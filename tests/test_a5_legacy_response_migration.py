"""A5 legacy response migration against actual program/stage/context evidence.

The r3 closure requires a ready rule to bind the actual KernelProgram/Stage/
RecipeContext occurrence, work/unit/owner/repetition, phase and fixed boundary.
Old term indices/formulas are diagnostics. Source roles from T06's
``training_sources`` decide identification; validation/diagnostic references do
not veto a new training basis. Replay remains observation-only.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.observations.resource_services import normalize_resource_service
from mindie_sim.observations.service_fitting import fit_resource_service
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.services import (ResourceService, ServiceCurve,
                                            service_identity)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.services import ProfileServiceRecipe
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import ProgramFragment, WorkItem, WorkSet
from mindie_sim.prediction.service_regimes import (
    assess_regime_migration,
    resource_phase_bindings,
)
from mindie_sim.profile_observed import _extract_observed, replay_regime_observations
from mindie_sim.schema.common import ModelingError
from test_mechanism_contract import hardware_case
from test_resource_services import copy_op, execute

ROOT = Path(__file__).resolve().parents[1]
A5_REAL = (
    ROOT / "data/observations/a5-grouped-service-regimes-20261030/parameters.json",
    ROOT / "data/observations/a5-grouped-service-regimes-20261030/grouped-parameters.json",
    ROOT / "data/observations/a5-mc2-service-regimes-20261030/parameters.json",
    ROOT / "data/observations/a5-mc2-service-regimes-20261030/int8-ep4-parameters.json",
    ROOT / "data/observations/a5-hccl-service-regimes-20261030/tp4-parameters.json",
    ROOT / "data/observations/a5-qwen35-ep1-grouped-20261030/parameters.json",
)
OLD_NEGATIVE = {
    "kind": "operator_response_calibration",
    "source": "old-capture",
    "resource_identifiability": "no independent hardware sweep: effective call "
                                "response is not assigned to a guessed HBM/AIV/link primitive",
}


def context(native, *, rates=None, owners=None, kernel=None, units=None):
    resources = list(native)
    units = units or {resource: "bytes" for resource in resources}
    names = {resource: f"{resource}@r0:s0" for resource in resources}
    rates = rates or {resource: 96000.0 for resource in resources}
    return RecipeContext(
        op=None, work={}, resource_names=names,
        rates={names[resource]: float(rates[resource]) for resource in resources},
        core_count=32, native_work=dict(native), work_units=dict(units),
        resource_units=dict(owners or {resource: 1 for resource in resources}),
        service_identity={"kernel": kernel} if kernel else {})


def stage(name, resource, amount, *, unit="bytes", after=(), repeats=1):
    fragment = ProgramFragment(
        name=name,
        work=WorkSet((WorkItem(resource=resource, amount=amount, unit=unit,
                              source="actual-lowering"),)),
        repeats=repeats)
    return Stage(name, work={resource: amount}, work_units={resource: unit},
                 after=tuple(after), fragment=fragment)


def hbm_program():
    return KernelProgram((stage("hbm", "hbm", 1536),))


def hbm_context(*, native=1536, owner_units=1, kernel=None):
    return context({"hbm": native}, owners={"hbm": owner_units}, kernel=kernel)


def two_stage_program():
    return KernelProgram((stage("hbm", "hbm", 1536),
                          stage("link", "link", 768, after=("hbm",))))


def two_stage_context(*, hbm=1536, link=768, owners=1):
    return context({"hbm": hbm, "link": link},
                   owners={"hbm": owners, "link": owners})


def native_program():
    return KernelProgram((stage("native", "native_call_service", 1536),))


def native_context(*, native=1536, kernel="NativeCall", owner_units=1):
    return context({"native_call_service": native},
                   owners={"native_call_service": owner_units}, kernel=kernel)


def affine_rule(**changes):
    rule = {
        "id": "hbm-copy-response",
        "match": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "coordinates": {"bytes": {"path": "work.hbm", "scale": 1}},
        "domain": {"bytes": {"min": 1, "max": 4000}},
        "response": {"fixed_us": 2, "terms": [{
            "coordinate": "bytes", "us_per_unit": 0.5,
            "scale_with": {"resource": "hbm", "reference_rate": 2e9}}]},
        "provenance": {"kind": "synthetic_causal_control"},
    }
    rule.update(changes)
    return rule


def two_term_rule():
    return affine_rule(response={"fixed_us": 2, "terms": [
        {"coordinate": "bytes", "us_per_unit": 0.5,
         "scale_with": {"resource": "hbm", "reference_rate": 2e9}},
        {"coordinate": "bytes", "us_per_unit": 0.25,
         "scale_with": {"resource": "link", "reference_rate": 1e9}}]})


def joint_rule():
    return {
        "id": "native-response",
        "match": {"kernel": "NativeCall"},
        "coordinates": {"bytes": {"path": "work.hbm"}},
        "domain": {"bytes": {"min": 1, "max": 100000}},
        "response": {"fixed_us": 2, "terms": [{
            "coordinate": "bytes", "us_per_unit": 0.5,
            "scale_with": {"resource": "native_call_service",
                           "reference_rate": 1e9}}]},
        "execution": {"kind": "joint_native", "resource": "native_call_service",
                      "reference_rate": 1e9, "native_kernel": "NativeCall"},
        "provenance": {"boundary": "inclusive device call"},
    }


def fixed_only_rule(**changes):
    rule = {
        "id": "fixed-only",
        "match": {"hardware": "Ascend950DT_9582"},
        "response": {"fixed_us": 2},
        "execution": {"kind": "joint_native", "resource": "native_call_service",
                      "reference_rate": 1e9, "native_kernel": "NativeCall"},
        "provenance": {"boundary": "inclusive device call"},
    }
    rule.update(changes)
    return rule


def service(**changes):
    raw = {
        "id": "hbm-copy-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "supply": {"kind": "absolute_rate", "rate": 96e6},
        "rate_scope": "aggregate", "active_units": 1,
        "conditions": {"hardware": "Ascend950DT_9582"},
        "identification": {
            "status": "identified", "resource_identified": True,
            "rate_source": "fitted", "fixed_source": "fitted",
            "fixed_identified": True, "train_work": [1536.0, 3072.0]},
        "provenance": {"kind": "calibrated_resource_service",
                       "training_sources": ["new-run-a", "new-run-b"]},
    }
    raw.update(changes)
    return normalize_resource_service(raw)


def link_service(**changes):
    raw = {
        "id": "link-copy-service", "scope": "independent_resource",
        "resources": ["link"], "phase": "copy", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "supply": {"kind": "absolute_rate", "rate": 96e6},
        "rate_scope": "aggregate", "active_units": 1,
        "conditions": {"hardware": "Ascend950DT_9582"},
        "identification": {
            "status": "identified", "resource_identified": True,
            "rate_source": "fitted", "fixed_source": "fitted",
            "fixed_identified": True, "train_work": [1536.0, 3072.0]},
        "provenance": {"kind": "calibrated_resource_service",
                       "training_sources": ["new-link-run-a", "new-link-run-b"]},
    }
    raw.update(changes)
    return normalize_resource_service(raw)


def joint_service(**changes):
    raw = {
        "id": "native-window-service", "scope": "joint_phase",
        "resources": ["native_call_service"], "phase": "native_call",
        "work_unit": "bytes", "boundary": "resource_phase",
        "startup_us": 2, "response_us": 0,
        "supply": {"kind": "absolute_rate", "rate": 96e6},
        "rate_scope": "aggregate", "active_units": 1,
        "conditions": {"regime_rule_id": "native-response",
                       "native_boundary": "NativeCall"},
        "identification": {
            "status": "joint_identified", "rate_source": "fitted",
            "fixed_source": "fitted", "fixed_identified": True,
            "train_work": [1536.0, 3072.0]},
        "provenance": {"kind": "joint_native_resource_phase",
                       "training_sources": ["new-joint-run-a", "new-joint-run-b"]},
    }
    raw.update(changes)
    return normalize_resource_service(raw)


def cube_service(**changes):
    raw = {
        "id": "cube-service", "scope": "independent_resource",
        "resources": ["cube"], "phase": "mac", "work_unit": "flops",
        "boundary": "resource_phase", "startup_us": 1, "response_us": 0,
        "supply": {"kind": "absolute_rate", "rate": 1e12},
        "rate_scope": "aggregate", "active_units": 1,
        "conditions": {},
        "identification": {
            "status": "identified", "resource_identified": True,
            "rate_source": "fitted", "fixed_source": "fitted",
            "fixed_identified": True, "train_work": [1000000.0, 2000000.0]},
        "provenance": {"kind": "calibrated_resource_service",
                       "training_sources": ["new-cube-run-a", "new-cube-run-b"]},
    }
    raw.update(changes)
    return normalize_resource_service(raw)


def plan(program=None, runtime=None, **changes):
    value = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": program or hbm_program(), "context": runtime or hbm_context(),
        "condition_context": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "fixed": [{"service_id": "hbm-copy-service", "stage": "hbm"}],
        "work": [{"term_index": 0, "service_id": "hbm-copy-service",
                  "stage": "hbm", "resource": "hbm", "work_unit": "bytes",
                  "multiplicity": 1, "phase": "copy"}],
    }
    value.update(changes)
    return value


def joint_plan(program=None, runtime=None, **changes):
    value = {
        "replacement_kind": "joint_phase_service", "scope": "complete_rule",
        "program": program or native_program(),
        "context": runtime or native_context(),
        "condition_context": {"kernel": "NativeCall",
                              "regime_rule_id": "native-response",
                              "native_boundary": "NativeCall"},
        "native_boundary": "NativeCall",
        "fixed": [{"service_id": "native-window-service", "stage": "native"}],
        "work": [{"term_index": 0, "service_id": "native-window-service",
                  "stage": "native", "resource": "native_call_service",
                  "work_unit": "bytes", "multiplicity": 1,
                  "phase": "native_call", "native_boundary": "NativeCall"}],
    }
    value.update(changes)
    return value


def assess_one(rule, services=(), binding=None):
    links = None if binding is None else {rule["id"]: binding}
    return assess_regime_migration(
        [rule], resource_services=services, links=links)["rules"][0]


def test_ready_requires_an_actual_program_binding():
    unlinked = assess_one(affine_rule())
    assert unlinked["status"] == "awaiting_resource_service"
    candidate = assess_regime_migration(
        [affine_rule()], resource_services=[service()],
        links={"hbm-copy-response": ["hbm-copy-service"]})["rules"][0]
    assert candidate["status"] == "awaiting_binding_plan"
    term_only = plan()
    term_only.pop("program")
    term_only.pop("context")
    term_only["fixed"] = []
    record = assess_one(affine_rule(), [service()], term_only)
    assert record["status"] == "awaiting_program_binding"
    assert "migration_program_binding_required" in record["blockers"]


def test_complete_single_resource_actual_binding_is_ready():
    record = assess_one(affine_rule(), [service()], plan())
    assert record["status"] == "resource_phase_ready"
    assert record["blockers"] == ["t04_executor_switch_pending"]
    coverage = record["coverage"]
    assert coverage["complete"] is True
    assert coverage["combination"] == "actual_program_dependencies"
    assert coverage["program"][0]["name"] == "hbm"
    assert coverage["program"][0]["after"] == []
    work = coverage["work_terms"][0]
    assert work["bound"] is True
    assert work["bindings"][0]["stage"] == "hbm"
    assert work["coordinate_declaration"]["path"] == "work.hbm"
    assert work["legacy_us_per_unit"] == pytest.approx(0.5)
    assert coverage["fixed"] == [{
        "service_id": "hbm-copy-service", "stage": "hbm", "phase": "copy",
        "new_fixed_us": 2.0, "native_boundary": ""}]
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["actual_work"] == pytest.approx(1536)
    assert rate["actual_unit"] == "bytes"
    assert rate["owner_units"] == 1
    assert rate["rate_used"] == pytest.approx(96e6)
    assert rate["amount_basis"] == "program_fragment_work"
    handoff = resource_phase_bindings(
        [affine_rule()], resource_services=[service()],
        links={"hbm-copy-response": plan()})
    assert handoff["production_switch_performed"] is False
    assert len(handoff["bindings"]) == 1
    assert handoff["bindings"][0]["coverage"]["complete"] is True


def test_old_rate_and_fixed_differences_are_diagnostic():
    record = assess_one(
        affine_rule(), [service(supply={"kind": "absolute_rate", "rate": 2.02e6},
                                startup_us=3)],
        plan())
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["rate_used"] == pytest.approx(2.02e6)
    assert rate["actual_work"] == pytest.approx(1536)
    assert record["diagnostics"]["fixed_comparison"] == {
        "legacy_fixed_us": 2.0, "service_fixed_total_us": 3.0,
        "difference_is_diagnostic": True}
    assert record["blockers"] == ["t04_executor_switch_pending"]


def test_old_negative_provenance_uses_new_training_and_allows_old_validation_ref():
    rule = affine_rule(provenance=OLD_NEGATIVE)
    new = service(provenance={
        "kind": "calibrated_resource_service",
        "training_sources": ["new-run-a"],
        "validation_sources": [{"source": "old-capture"}],
    })
    record = assess_one(rule, [new], plan())
    assert record["status"] == "resource_phase_ready"
    assert record["provenance_evidence"][0]["key"] == "resource_identifiability"
    assert record["diagnostics"]["old_provenance_negative"] is True
    assert record["service_evidence"]["hbm-copy-service"]["source_roles"]["training"] == [
        "new-run-a"]
    assert record["service_evidence"]["hbm-copy-service"]["source_roles"][
        "validation"] == ["old-capture"]
    assert not any("provenance" in item or "training" in item
                   for item in record["blockers"])

    old_training = service(
        provenance={"kind": "calibrated_resource_service",
                    "training_sources": ["old-capture"]},
        identification={"status": "identified", "resource_identified": True})
    record = assess_one(rule, [old_training], plan())
    assert record["status"] == "observation_replay"
    assert any("replacement_rate_not_independently_identified" in item
               or "replacement_training_depends_on_unidentifiable_legacy_source"
               in item for item in record["blockers"])
    shared = service(provenance={
        "kind": "calibrated_resource_service",
        "training_sources": ["old-capture", "new-run-a", "new-run-b"]})
    record = assess_one(rule, [shared], plan())
    assert record["status"] == "resource_phase_ready"
    assert "old-capture" in record["service_evidence"][
        "hbm-copy-service"]["legacy_training_overlap"]

    no_source = service(provenance={"kind": "calibrated_resource_service"})
    record = assess_one(rule, [no_source], plan())
    assert record["status"] == "observation_replay"
    assert any("replacement_service_requires_training_sources" in item
               for item in record["blockers"])

    replay = assess_one(rule)
    assert replay["status"] == "observation_replay"


def test_t06_fit_training_sources_and_h5_rates_are_consumed():
    def fitted(rate_us, duration_1536, duration_3072):
        packet = {
            "id": "hbm-copy-service", "scope": "independent_resource",
            "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
            "boundary": "resource_phase", "rate_scope": "aggregate",
            "active_units": 1, "conditions": {"hardware": "Ascend950DT_9582"},
            "provenance": {"window": "resource_phase"},
            "samples": [
                {"work": 1536, "duration_us": duration_1536, "run_id": "new-a",
                 "repeat_index": 0, "split": "train", "source": "new-run-a"},
                {"work": 3072, "duration_us": duration_3072, "run_id": "new-b",
                 "repeat_index": 0, "split": "train", "source": "new-run-b"}],
        }
        result = fit_resource_service(packet)
        assert result["service"]["provenance"]["training_sources"] == [
            "new-run-a", "new-run-b"]
        return result["service"]

    packet = {
        "id": "hbm-copy-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "conditions": {"hardware": "Ascend950DT_9582"},
        "provenance": {"window": "resource_phase"},
        "samples": [
            {"work": 1536, "duration_us": 18, "run_id": "new-a",
             "repeat_index": 0, "split": "train", "source": "new-run-a"},
            {"work": 3072, "duration_us": 34, "run_id": "new-b",
             "repeat_index": 0, "split": "train", "source": "new-run-b"},
            {"work": 4608, "duration_us": 50, "run_id": "old-c",
             "repeat_index": 0, "split": "train", "source": "old-capture"}],
    }
    shared_fit = fit_resource_service(packet)["service"]
    assert shared_fit["provenance"]["training_sources"] == [
        "new-run-a", "new-run-b", "old-capture"]
    record = assess_one(affine_rule(provenance=OLD_NEGATIVE), [shared_fit], plan())
    assert record["status"] == "resource_phase_ready"
    assert "old-capture" in record["service_evidence"][
        "hbm-copy-service"]["legacy_training_overlap"]

    slow = fitted(96, 18, 34)
    fast = fitted(128, 14, 26)
    assert normalize_resource_service(slow).duration_us(1536) == pytest.approx(18)
    assert normalize_resource_service(fast).duration_us(1536) == pytest.approx(14)
    for service_record, expected in ((slow, 18), (fast, 14)):
        record = assess_one(affine_rule(), [service_record], plan())
        assert record["status"] == "resource_phase_ready"
        rate = record["diagnostics"]["rate_comparison"][0]
        assert rate["actual_work"] == pytest.approx(1536)
        assert rate["actual_unit"] == "bytes"
        assert rate["rate_used"] == pytest.approx(
            normalize_resource_service(service_record).effective_rate)
        assert normalize_resource_service(service_record).duration_us(1536) == (
            pytest.approx(expected))
        assert record["coverage"]["fixed_total_us"] == pytest.approx(2)
        larger = assess_one(
            affine_rule(), [service_record],
            plan(program=KernelProgram((stage("hbm", "hbm", 3072),)),
                 runtime=hbm_context(native=3072)))
        assert larger["status"] == "resource_phase_ready"
        assert larger["diagnostics"]["rate_comparison"][0]["actual_work"] == (
            pytest.approx(3072))
        assert normalize_resource_service(service_record).duration_us(3072) == (
            pytest.approx(34 if expected == 18 else 26))
        unseen = assess_one(
            affine_rule(), [service_record],
            plan(program=KernelProgram((stage("hbm", "hbm", 2304),)),
                 runtime=hbm_context(native=2304)))
        assert unseen["status"] == "resource_phase_ready"
        assert unseen["diagnostics"]["rate_comparison"][0]["actual_work"] == (
            pytest.approx(2304))
        assert normalize_resource_service(service_record).duration_us(2304) == (
            pytest.approx(26 if expected == 18 else 20))


def test_external_occurrence_matches_builtin_fragment():
    builtin = assess_one(affine_rule(), [service()], plan())
    external_stage = Stage("hbm", work={"hbm": 1536},
                           work_units={"hbm": "bytes"})
    external_fragment = ProgramFragment(
        name="external-hbm",
        work=WorkSet((WorkItem(resource="hbm", amount=1536,
                              unit="bytes", source="actual-lowering"),)),
        repeats=1)
    external = assess_one(
        affine_rule(), [service()],
        plan(program=KernelProgram((external_stage,)),
             occurrences={"hbm": external_fragment}))
    assert external["status"] == "resource_phase_ready"
    assert external["diagnostics"]["rate_comparison"][0]["actual_work"] == (
        pytest.approx(1536))
    assert external["diagnostics"]["rate_comparison"][0]["instance"] == (
        builtin["diagnostics"]["rate_comparison"][0]["instance"])
    assert service().duration_us(1536) == pytest.approx(18)


def test_builtin_and_external_fragments_are_normalized_once():
    same = ProgramFragment(
        name="external-equivalent",
        work=WorkSet((WorkItem(resource="hbm", amount=1536,
                              unit="bytes", source="actual-lowering"),)),
        repeats=1)
    record = assess_one(
        affine_rule(), [service()],
        plan(program=hbm_program(), occurrences={"hbm": same}))
    assert record["status"] == "resource_phase_ready"
    assert len(record["coverage"]["actual_work_instances"]) == 1
    assert record["diagnostics"]["rate_comparison"][0]["actual_work"] == (
        pytest.approx(1536))
    conflicting = ProgramFragment(
        name="external-conflict",
        work=WorkSet((WorkItem(resource="hbm", amount=768,
                              unit="bytes", source="actual-lowering"),)),
        repeats=1)
    with pytest.raises(ModelingError,
                       match="migration_plan_occurrence_conflicts_with_builtin_fragment"):
        assess_one(affine_rule(), [service()],
                   plan(program=hbm_program(), occurrences={"hbm": conflicting}))


def test_positive_link_work_without_fragment_is_undelivered():
    program = KernelProgram((
        stage("hbm", "hbm", 1536),
        Stage("link", work={"link": 768}, work_units={"link": "bytes"})))
    binding = plan(
        program=program, runtime=two_stage_context(),
        fixed=[{"service_id": "hbm-copy-service", "stage": "hbm"},
               {"service_id": "link-copy-service", "stage": "link"}],
        work=[
            {"term_index": 0, "service_id": "hbm-copy-service",
             "stage": "hbm", "resource": "hbm", "work_unit": "bytes",
             "multiplicity": 1, "phase": "copy"},
            {"term_index": 1, "service_id": "link-copy-service",
             "stage": "link", "resource": "link", "work_unit": "bytes",
             "multiplicity": 1, "phase": "copy"}])
    record = assess_one(two_term_rule(), [service(), link_service()], binding)
    assert record["status"] == "partial"
    assert record["coverage"]["complete"] is False
    assert record["coverage"]["undelivered_work"]
    assert any("actual_work_undelivered:link" in item
               for item in record["blockers"])
    handoff = resource_phase_bindings(
        [two_term_rule()], resource_services=[service(), link_service()],
        links={"hbm-copy-response": binding})
    assert handoff["bindings"] == []


def test_complete_program_without_term_index_is_ready():
    binding = plan(work=[{
        "service_id": "hbm-copy-service", "stage": "hbm",
        "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
        "phase": "copy"}])
    record = assess_one(affine_rule(), [service()], binding)
    assert record["status"] == "resource_phase_ready"
    assert record["coverage"]["complete"] is True
    assert all(row.get("diagnostic_only") is True
               for row in record["coverage"]["work_terms"])
    assert record["diagnostics"]["rate_comparison"][0]["actual_work"] == (
        pytest.approx(1536))


def test_single_owner_placement_overrides_device_owner_count():
    physical = "hbm@r0:s0"
    fragment = ProgramFragment(
        name="hbm-owner",
        work=WorkSet((WorkItem(resource=physical, amount=1536,
                              unit="bytes", source="actual-lowering"),)),
        repeats=1)
    owned_stage = Stage("hbm", work={physical: 1536},
                        work_units={physical: "bytes"},
                        placement={"hbm": ("r0",)}, fragment=fragment)
    runtime = RecipeContext(
        op=None, work={}, resource_names={"hbm": physical},
        rates={physical: 96000.0}, core_count=32,
        native_work={"hbm": 1536}, work_units={"hbm": "bytes"},
        resource_units={"hbm": 2}, service_identity={})
    program = KernelProgram((owned_stage,))
    two_owner = service(rate_scope="per_owner", active_units=2,
                        supply={"kind": "absolute_rate", "rate": 48e6})
    binding = plan(program=program, runtime=runtime,
                   work=[{"term_index": 0, "service_id": "hbm-copy-service",
                          "stage": "hbm", "resource": "hbm",
                          "work_unit": "bytes", "multiplicity": 1,
                          "phase": "copy", "work_scope": "per_owner"}])
    record = assess_one(affine_rule(), [two_owner], binding)
    assert record["status"] == "conflict"
    assert any("per_owner_active_units_do_not_match_actual_owner_units" in item
               for item in record["blockers"])
    one_owner = service(rate_scope="per_owner", active_units=1,
                        supply={"kind": "absolute_rate", "rate": 48e6})
    record = assess_one(affine_rule(), [one_owner], binding)
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["owner_units"] == 1
    assert rate["actual_work"] == pytest.approx(1536)
    assert 2 + rate["actual_work"] * 1e6 / rate["rate_used"] == pytest.approx(34)


def test_t06_declared_fixed_single_point_identifies_rate():
    packet = {
        "id": "hbm-copy-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "conditions": {"hardware": "Ascend950DT_9582"},
        "startup_us": 2,
        "samples": [{"work": 1536, "duration_us": 18, "run_id": "new-a",
                     "repeat_index": 0, "split": "train",
                     "source": "new-run-a"}]}
    declared = fit_resource_service(packet)["service"]
    assert declared["identification"]["rate_source"] == "fitted_with_declared_fixed"
    assert declared["identification"]["fixed_source"] == "declared"
    assert declared["identification"]["train_work"] == [1536.0]
    record = assess_one(affine_rule(), [declared], plan())
    assert record["status"] == "resource_phase_ready"
    assert record["service_evidence"]["hbm-copy-service"]["fixed_source"] == "declared"
    assert normalize_resource_service(declared).duration_us(1536) == pytest.approx(18)
    larger = assess_one(
        affine_rule(), [declared],
        plan(program=KernelProgram((stage("hbm", "hbm", 2304),)),
             runtime=hbm_context(native=2304)))
    assert larger["status"] == "resource_phase_ready"
    assert normalize_resource_service(declared).duration_us(2304) == pytest.approx(26)


def test_physical_resource_key_maps_to_logical_role():
    fragment = ProgramFragment(
        name="hbm-read",
        work=WorkSet((WorkItem(resource="hbm@r0:s0", amount=1536,
                              unit="bytes", source="actual-lowering"),)),
        repeats=1)
    physical_stage = Stage("hbm", work={"hbm@r0:s0": 1536},
                           work_units={"hbm@r0:s0": "bytes"},
                           fragment=fragment)
    runtime = RecipeContext(
        op=None, work={}, resource_names={"hbm": "hbm@r0:s0"},
        rates={"hbm@r0:s0": 96000.0}, core_count=32,
        native_work={"hbm": 1536}, work_units={"hbm": "bytes"},
        resource_units={"hbm": 1}, service_identity={})
    record = assess_one(
        affine_rule(), [service()],
        plan(program=KernelProgram((physical_stage,)), runtime=runtime))
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["actual_work"] == pytest.approx(1536)
    assert rate["instance"][1] == "hbm@r0:s0"
    assert rate["charged_once"] is True


def test_two_old_terms_on_hbm_leave_actual_link_uncovered():
    binding = plan(program=two_stage_program(), runtime=two_stage_context(),
                   fixed=[{"service_id": "hbm-copy-service", "stage": "hbm"}],
                   work=[
                       {"term_index": 0, "service_id": "hbm-copy-service",
                        "stage": "hbm", "resource": "hbm",
                        "work_unit": "bytes", "multiplicity": 1, "phase": "copy"},
                       {"term_index": 1, "service_id": "hbm-copy-service",
                        "stage": "hbm", "resource": "hbm",
                        "work_unit": "bytes", "multiplicity": 1, "phase": "copy"}])
    record = assess_one(two_term_rule(), [service()], binding)
    assert record["status"] != "resource_phase_ready"
    assert record["coverage"]["complete"] is False
    assert any("actual_work_uncovered:link" in item for item in record["blockers"])
    assert [row["bound"] for row in record["coverage"]["work_terms"]] == [True, True]
    assert all(row["charged_once"] is True
               for row in record["diagnostics"]["rate_comparison"])
    assert record["coverage"]["fixed_total_us"] == pytest.approx(2)


def test_two_copy_instances_on_shared_resource_are_each_bound_once():
    rule = affine_rule(response={"fixed_us": 0, "terms": [
        {"coordinate": "bytes", "us_per_unit": 0.5,
         "scale_with": {"resource": "hbm", "reference_rate": 2e9}},
        {"coordinate": "bytes", "us_per_unit": 0.25,
         "scale_with": {"resource": "hbm", "reference_rate": 2e9}}]})
    program = KernelProgram((stage("copy1", "hbm", 1536),
                             stage("copy2", "hbm", 1536)))
    runtime = context({"hbm": 3072}, owners={"hbm": 1})
    binding = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": program, "context": runtime,
        "condition_context": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "fixed": [],
        "work": [
            {"term_index": 0, "service_id": "hbm-copy-service", "stage": "copy1",
             "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
             "phase": "copy"},
            {"term_index": 1, "service_id": "hbm-copy-service", "stage": "copy2",
             "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
             "phase": "copy"}]}
    record = assess_one(rule, [service(startup_us=0)], binding)
    assert record["status"] == "resource_phase_ready"
    assert len(record["coverage"]["actual_work_instances"]) == 2
    assert record["coverage"]["fixed_total_us"] == pytest.approx(0)
    rates = record["diagnostics"]["rate_comparison"]
    assert [rate["actual_work"] for rate in rates] == [1536, 1536]
    assert all(rate["charged_once"] is True for rate in rates)
    shared = service(startup_us=0)
    assert shared.duration_us(3072) == pytest.approx(32)
    assert shared.duration_us(1536) == pytest.approx(16)


def test_real_zero_work_cannot_be_bound_to_a_service():
    zero_program = KernelProgram((stage("hbm", "hbm", 0),))
    record = assess_one(affine_rule(), [service()],
                        plan(program=zero_program))
    assert record["status"] != "resource_phase_ready"
    assert any("service_bound_for_real_zero_work" in item
               for item in record["blockers"])
    assert record["coverage"]["real_zero_work"] == []


def test_fixed_link_startup_cannot_stand_for_hbm_stage():
    work = service(startup_us=0)
    fixed_link = link_service()
    binding = plan(fixed=[{"service_id": "link-copy-service", "stage": "hbm"}],
                   work=[{"term_index": 0, "service_id": "hbm-copy-service",
                          "stage": "hbm", "resource": "hbm",
                          "work_unit": "bytes", "multiplicity": 1,
                          "phase": "copy"}])
    record = assess_one(affine_rule(), [work, fixed_link], binding)
    assert record["status"] == "conflict"
    assert any("fixed_service_resource_not_on_actual_stage" in item
               for item in record["blockers"])


def test_partial_actual_program_coverage_keeps_completed_slice():
    binding = plan(program=two_stage_program(), runtime=two_stage_context())
    binding["work"] = binding["work"][:1]
    record = assess_one(two_term_rule(), [service()], binding)
    assert record["status"] == "partial"
    assert [(row["term_index"], row["bound"], row["unbound"])
            for row in record["coverage"]["work_terms"]] == [
                (0, True, False), (1, False, True)]
    assert "legacy_work_term_unbound:1" in record["blockers"]
    assert record["coverage"]["work_terms"][1]["legacy_us_per_unit"] == pytest.approx(0.25)


def test_valid_and_invalid_actual_binding_cannot_coexist_as_ready():
    binding = plan(program=two_stage_program(), runtime=two_stage_context(), work=[
        {"term_index": 0, "service_id": "hbm-copy-service", "stage": "hbm",
         "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
         "phase": "copy"},
        {"term_index": 1, "service_id": "link-copy-service", "stage": "link",
         "resource": "link", "work_unit": "bytes", "multiplicity": 1,
         "phase": "copy"}])
    bad_link = link_service(resources=["hbm"])
    record = assess_one(two_term_rule(), [service(), bad_link], binding)
    assert record["status"] == "conflict"
    rows = {row["term_index"]: row for row in record["coverage"]["work_terms"]}
    assert rows[0]["bound"] is True
    assert rows[1]["bound"] is False
    assert any("actual_stage_resource_not_covered_by_service" in item
               for item in record["blockers"])


def test_duplicate_term_conflicts_and_multiplicity_is_preserved():
    duplicate = plan(work=[
        {"term_index": 0, "service_id": "hbm-copy-service", "stage": "hbm",
         "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
         "phase": "copy"},
        {"term_index": 0, "service_id": "hbm-copy-service", "stage": "hbm",
         "resource": "hbm", "work_unit": "bytes", "multiplicity": 2,
         "phase": "copy"}])
    assert assess_one(affine_rule(), [service()], duplicate)["status"] == "conflict"
    fabricated = plan(work=[
        {"term_index": 0, "service_id": "hbm-copy-service", "stage": "hbm",
         "resource": "hbm", "work_unit": "bytes", "multiplicity": 3,
         "phase": "copy"}])
    record = assess_one(affine_rule(), [service()], fabricated)
    assert record["status"] == "conflict"
    assert any("multiplicity_does_not_match_actual_occurrence" in item
               for item in record["blockers"])
    repeated = plan(
        program=KernelProgram((stage("hbm", "hbm", 1536, repeats=3),)),
        runtime=hbm_context(native=4608),
        work=[{"term_index": 0, "service_id": "hbm-copy-service",
               "stage": "hbm", "resource": "hbm", "work_unit": "bytes",
               "multiplicity": 3, "phase": "copy"}])
    record = assess_one(affine_rule(), [service()], repeated)
    assert record["status"] == "resource_phase_ready"
    assert record["coverage"]["work_terms"][0]["bindings"][0]["multiplicity"] == 3
    assert record["diagnostics"]["rate_comparison"][0]["actual_work"] == pytest.approx(
        4608)


def test_missing_typed_condition_is_unresolved_not_ready():
    service_dtype = service(conditions={
        "hardware": "Ascend950DT_9582", "dtype": "bf16"})
    record = assess_one(affine_rule(), [service_dtype], plan())
    assert record["status"] == "awaiting_conditions"
    assert any("condition_context_missing:dtype" in item for item in record["blockers"])
    checks = {check["path"]: check for check in record["conditions"][
        "services"]["hbm-copy-service"]["checks"]}
    assert checks["hardware"]["matched"] is True
    assert checks["dtype"]["present"] is False


def test_extra_service_condition_requires_an_explicit_slice():
    service_dtype = service(conditions={
        "hardware": "Ascend950DT_9582", "dtype": "bf16"})
    complete = plan(condition_context={
        "hardware": "Ascend950DT_9582", "recipe": "copy", "dtype": "bf16"})
    record = assess_one(affine_rule(), [service_dtype], complete)
    assert record["status"] == "conflict"
    assert any("service_condition_outside_rule_scope" in item
               for item in record["blockers"])
    sliced = plan(scope="slice", slice_conditions={"dtype": "bf16"},
                  condition_context={"hardware": "Ascend950DT_9582",
                                     "recipe": "copy", "dtype": "bf16"})
    record = assess_one(affine_rule(), [service_dtype], sliced)
    assert record["status"] == "slice_ready"
    assert record["scope"] == {"kind": "slice",
                               "slice_conditions": {"dtype": "bf16"}}
    assert record["coverage"]["complete"] is True


@pytest.mark.parametrize("change, blocker", (
    ({"work_unit": "rows"}, "binding_work_unit_does_not_match_actual_stage"),
    ({"phase": "other"}, "binding_phase_does_not_match_service"),
    ({"work_scale": 0}, "requires_positive_rate"),
))
def test_work_binding_requires_actual_unit_and_phase(change, blocker):
    item = {"term_index": 0, "service_id": "hbm-copy-service", "stage": "hbm",
            "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
            "phase": "copy"}
    item.update(change)
    binding = plan(work=[item])
    if "requires_positive" in blocker:
        with pytest.raises(ModelingError, match=blocker):
            assess_one(affine_rule(), [service()], binding)
        return
    record = assess_one(affine_rule(), [service()], binding)
    assert record["status"] == "conflict"
    assert any(blocker in text for text in record["blockers"])


def test_native_boundary_mismatch_with_actual_program_is_rejected():
    record = assess_one(joint_rule(), [joint_service()], joint_plan())
    assert record["status"] == "resource_phase_ready"
    assert record["diagnostics"]["actual_native_boundary"] == "NativeCall"
    wrong = joint_service(conditions={"regime_rule_id": "native-response",
                                      "native_boundary": "OtherCall"})
    binding = joint_plan(native_boundary="OtherCall")
    binding["work"][0]["native_boundary"] = "OtherCall"
    binding["condition_context"]["native_boundary"] = "OtherCall"
    record = assess_one(joint_rule(), [wrong], binding)
    assert record["status"] == "conflict"
    assert any("actual_native_boundary_conflict" in item or
               "native_boundary_does_not_match_actual_program" in item
               for item in record["blockers"])
    missing = joint_service(conditions={"regime_rule_id": "native-response"})
    record = assess_one(joint_rule(), [missing], joint_plan())
    assert record["status"] == "conflict"
    assert any("joint_service_requires_native_boundary" in item
               for item in record["blockers"])


def test_unrelated_fixed_and_two_stage_each_own_startup():
    unrelated = service(id="other-fixed", resources=["hbm"], phase="unrelated",
                        startup_us=2)
    work = service(startup_us=0)
    binding = plan(fixed=[{"service_id": "other-fixed", "stage": "hbm"}])
    record = assess_one(affine_rule(), [work, unrelated], binding)
    assert record["status"] == "conflict"
    assert any("fixed_phase_does_not_match_stage_work" in item
               for item in record["blockers"])

    binding = plan(program=two_stage_program(), runtime=two_stage_context(),
                   fixed=[{"service_id": "hbm-copy-service", "stage": "hbm"},
                          {"service_id": "link-copy-service", "stage": "link"}],
                   work=[
                       {"term_index": 0, "service_id": "hbm-copy-service",
                        "stage": "hbm", "resource": "hbm",
                        "work_unit": "bytes", "multiplicity": 1, "phase": "copy"},
                       {"term_index": 1, "service_id": "link-copy-service",
                        "stage": "link", "resource": "link",
                        "work_unit": "bytes", "multiplicity": 1, "phase": "copy"}])
    record = assess_one(two_term_rule(), [service(), link_service()], binding)
    assert record["status"] == "resource_phase_ready"
    assert len(record["coverage"]["fixed"]) == 2
    assert record["coverage"]["fixed_total_us"] == pytest.approx(4)
    assert [row["stage"] for row in record["coverage"]["fixed"]] == ["hbm", "link"]


def test_fixed_only_replacement_needs_real_work_reference():
    native = service(id="native-window-service",
                     resources=["native_call_service"], phase="native_call",
                     conditions={})
    base = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": native_program(), "context": native_context(),
        "condition_context": {"hardware": "Ascend950DT_9582"},
        "fixed": [{"service_id": "native-window-service", "stage": "native"}]}
    with_work = {**base, "work": [{
        "service_id": "native-window-service", "stage": "native",
        "resource": "native_call_service", "work_unit": "bytes",
        "multiplicity": 1, "phase": "native_call"}]}
    record = assess_one(fixed_only_rule(), [native], with_work)
    assert record["status"] == "resource_phase_ready"
    assert record["coverage"]["external_work"][0]["bound"] is True
    record = assess_one(fixed_only_rule(), [native], base)
    assert record["status"] == "observation_replay"
    assert "actual_stage_work_reference_required" in record["blockers"]


def test_surface_and_measured_rules_can_use_real_work_reference():
    surface_rule = affine_rule(
        coordinates={"cat": {"path": "metadata.cat"}},
        domain={"cat": {"min": 0, "max": 10}},
        response={"kind": "piecewise_linear", "coordinate": "cat",
                  "knots": [[0, 10], [10, 20]]})
    surface_plan = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": hbm_program(), "context": hbm_context(),
        "condition_context": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "fixed": [{"service_id": "hbm-copy-service", "stage": "hbm"}],
        "work": [{"service_id": "hbm-copy-service", "stage": "hbm",
                  "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
                  "phase": "copy"}]}
    record = assess_one(surface_rule, [service()], surface_plan)
    assert record["status"] == "resource_phase_ready"
    assert record["coverage"]["external_work"][0]["bound"] is True
    assert assess_one(surface_rule)["status"] == "observation_replay"

    measured = affine_rule(
        coordinates={"m": {"path": "local_shape.0"},
                     "n": {"path": "local_shape.1"},
                     "k": {"path": "local_shape.2"}},
        response={"kind": "measured_dual_load", "fixed_us": 1,
                  "resource": "cube", "dtype": "bf16",
                  "loads": [
                      {"id": "saturated", "active_owners": 24,
                       "rate_flops_per_s": 1e12, "measured_issued_flops": 1000},
                      {"id": "underfilled", "active_owners": 1,
                       "rate_flops_per_s": 1e11, "measured_issued_flops": 100}],
                  "issue": {"native_type": "QBMM", "dtype": "bf16", "n": 16,
                            "k": 16, "m_alignment": 16},
                  "owner_grid": {"tile_m": 16, "tile_n": 16, "cores": 24}},
        execution={"kind": "mac_service", "resource": "cube",
                   "native_kernel": "QBMM", "tile_m": 16, "tile_n": 16,
                   "tile_k": 16, "slots": 1},
        domain={}, provenance={"boundary": "inclusive native kernel"})
    measured_plan = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": KernelProgram((stage("cube", "cube", 1_000_000, unit="flops"),)),
        "context": context({"cube": 1_000_000}, units={"cube": "flops"},
                           rates={"cube": 1e9}),
        "condition_context": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "fixed": [{"service_id": "cube-service", "stage": "cube"}],
        "work": [{"service_id": "cube-service", "stage": "cube",
                  "resource": "cube", "work_unit": "flops", "multiplicity": 1,
                  "phase": "mac"}]}
    record = assess_one(measured, [cube_service()], measured_plan)
    assert record["status"] == "resource_phase_ready"
    assert record["diagnostics"]["rate_comparison"][0]["actual_unit"] == "flops"


def test_per_owner_and_aggregate_scope_is_consumed():
    aggregate = assess_one(affine_rule(), [service()], plan())
    assert aggregate["status"] == "resource_phase_ready"
    rate = aggregate["diagnostics"]["rate_comparison"][0]
    assert rate["work_scope"] == "aggregate"
    assert rate["actual_work"] == pytest.approx(1536)
    assert rate["rate_used"] == pytest.approx(96e6)

    per_owner = service(rate_scope="per_owner", active_units=2,
                        supply={"kind": "absolute_rate", "rate": 48e6})
    binding = plan(runtime=hbm_context(owner_units=2))
    binding["work"][0]["work_scope"] = "per_owner"
    record = assess_one(affine_rule(), [per_owner], binding)
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["work_scope"] == "per_owner"
    assert rate["actual_work"] == pytest.approx(768)
    assert rate["aggregate_work"] == pytest.approx(1536)
    assert rate["owner_units"] == 2
    assert rate["rate_used"] == pytest.approx(48e6)
    assert 2 + rate["actual_work"] * 1e6 / rate["rate_used"] == pytest.approx(18)

    mismatch = assess_one(affine_rule(), [per_owner], plan())
    assert mismatch["status"] == "conflict"
    assert any("per_owner_active_units_do_not_match_actual_owner_units" in item
               for item in mismatch["blockers"])


def test_device_kernel_boundary_is_consumed_with_actual_stage():
    record = assess_one(affine_rule(), [service(boundary="device_kernel")], plan())
    assert record["status"] == "resource_phase_ready"
    evidence = record["service_evidence"]["hbm-copy-service"]
    assert evidence["boundary"] == "device_kernel"
    assert evidence["valid"] is True


def test_resource_phase_bindings_requires_a_complete_actual_binding():
    handoff = resource_phase_bindings(
        [affine_rule()], resource_services=[service()],
        links={"hbm-copy-response": plan()})
    assert len(handoff["bindings"]) == 1
    assert handoff["bindings"][0]["coverage"]["complete"] is True
    roofline = affine_rule(response={
        "kind": "roofline", "services": [{
            "coordinate": "bytes", "resource": "hbm", "utilization": 1.0}]})
    no_plan = assess_one(roofline)
    assert no_plan["status"] == "resource_phase_ready"
    assert no_plan["coverage"] is None
    handoff = resource_phase_bindings([roofline])
    assert handoff["bindings"] == []


def test_fixed_component_is_owned_once_and_required_when_declared():
    no_fixed = plan(fixed=[])
    record = assess_one(affine_rule(), [service()], no_fixed)
    assert record["status"] == "conflict"
    assert "legacy_fixed_component_unbound" in record["blockers"]
    zero_fixed = affine_rule(response={"fixed_us": 0, "terms": [{
        "coordinate": "bytes", "us_per_unit": 0.5,
        "scale_with": {"resource": "hbm", "reference_rate": 2e9}}]})
    record = assess_one(zero_fixed, [service()], plan())
    assert record["status"] == "resource_phase_ready"
    assert record["coverage"]["fixed_total_us"] == pytest.approx(2)
    assert record["coverage"]["legacy_fixed_us"] == pytest.approx(0)


def test_replay_returns_observations_without_residual_pricing():
    rule = affine_rule()
    result = replay_regime_observations([rule], [{
        "rule_id": "hbm-copy-response", "duration_us": 3.5,
        "coordinates": {"bytes": 3000}, "source": "unit-observation"}])
    record = result["records"][0]
    assert result["production_prediction"] is False
    assert record["role"] == "diagnostic_observed_replay"
    assert record["production_prediction_input"] is False
    assert record["observed_us"] == pytest.approx(3.5)
    assert record["declared_response_us"] == pytest.approx(1502.0)
    assert not any("residual" in key or "correction" in key for key in record)


def test_observed_profile_records_are_marked_diagnostic():
    observed = _extract_observed({"observed_step_ms": 9.0})
    assert observed["observation_role"] == "diagnostic_observed_replay"
    assert _extract_observed({"unrelated": True}) == {}


def test_real_a5_declarations_remain_replay_without_actual_bindings():
    total = 0
    for path in A5_REAL:
        assert path.is_file(), path
        declarations = json.loads(path.read_text(encoding="utf-8"))["service_regimes"]
        result = assess_regime_migration(declarations)
        total += result["summary"]["rules"]
        assert result["summary"]["resource_phase_ready"] == 0, path
        assert result["summary"]["slice_ready"] == 0, path
        assert result["summary"]["awaiting_resource_service"] == 0, path
        assert all(record["status"] == "observation_replay"
                   for record in result["rules"]), path
    assert total == 15


def test_invalid_plan_structure_is_rejected():
    for binding, match in (
            ({"scope": "complete_rule"}, "replacement_kind"),
            ({"replacement_kind": "independent_service"}, "requires_scope"),
            ({"replacement_kind": "independent_service", "scope": "slice"},
             "slice_requires_conditions"),
            ({"replacement_kind": "independent_service", "scope": "complete_rule",
              "slice_conditions": {"dtype": "bf16"}},
             "does_not_accept_slice_conditions"),
            (plan(work=[{"term_index": 99, "service_id": "hbm-copy-service",
                         "stage": "hbm", "resource": "hbm",
                         "work_unit": "bytes", "multiplicity": 1,
                         "phase": "copy"}]), "declared_term_index")):
        with pytest.raises(ModelingError, match=match):
            assess_one(affine_rule(), [service()], binding)


def _owned_physical_stage(*, fragment=True, placement=None):
    physical = "hbm@r0:s0"
    if fragment:
        work = WorkSet((WorkItem(resource=physical, amount=1536,
                                 unit="bytes", source="actual-lowering"),))
        built = ProgramFragment(name="hbm-owner", work=work, repeats=1)
    else:
        built = None
    return Stage("hbm", work={physical: 1536},
                 work_units={physical: "bytes"},
                 placement=placement if placement is not None else {physical: (0,)},
                 fragment=built)


def _owned_physical_context():
    physical = "hbm@r0:s0"
    return RecipeContext(
        op=None, work={}, resource_names={"hbm": physical},
        rates={physical: 96000.0}, core_count=32,
        native_work={"hbm": 1536}, work_units={"hbm": "bytes"},
        resource_units={"hbm": 2}, service_identity={})


def _owned_binding(program, runtime, *, scope="per_owner"):
    return plan(
        program=program, runtime=runtime,
        work=[{"term_index": 0, "service_id": "hbm-copy-service",
               "stage": "hbm", "resource": "hbm", "work_unit": "bytes",
               "multiplicity": 1, "phase": "copy", "work_scope": scope}])


def test_physical_placement_key_resolves_to_one_actual_owner():
    """The review's single-owner counterexample uses the physical placement key."""
    owned = service(rate_scope="per_owner", active_units=1,
                    supply={"kind": "absolute_rate", "rate": 48e6})
    binding = _owned_binding(KernelProgram((_owned_physical_stage(),)),
                             _owned_physical_context())
    two_owner = service(rate_scope="per_owner", active_units=2,
                        supply={"kind": "absolute_rate", "rate": 48e6})
    record = assess_one(affine_rule(), [two_owner], binding)
    assert record["status"] == "conflict"
    assert any("per_owner_active_units_do_not_match_actual_owner_units" in item
               for item in record["blockers"])

    record = assess_one(affine_rule(), [owned], binding)
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["owner_units"] == 1
    assert rate["actual_work"] == pytest.approx(1536)
    assert 2 + rate["actual_work"] * 1e6 / rate["rate_used"] == pytest.approx(34)
    handoff = resource_phase_bindings(
        [affine_rule()], resource_services=[owned],
        links={"hbm-copy-response": binding})
    assert len(handoff["bindings"]) == 1


@pytest.mark.parametrize("work_scope", ("aggregate", "per_owner"))
@pytest.mark.parametrize("term_index", (0, None))
def test_missing_fragment_is_partial_before_rate_arithmetic(work_scope, term_index):
    """A missing work instance never reaches aggregate or per-owner division."""
    service_record = (service() if work_scope == "aggregate"
                      else service(rate_scope="per_owner", active_units=1,
                                   supply={"kind": "absolute_rate", "rate": 48e6}))

    def binding_for(program):
        value = _owned_binding(program, _owned_physical_context(),
                               scope=work_scope)
        if term_index is None:
            value["work"][0].pop("term_index")
        return value

    binding = binding_for(KernelProgram((
        _owned_physical_stage(fragment=False, placement={"hbm": ("r0",)}),)))
    record = assess_one(affine_rule(), [service_record], binding)
    assert record["status"] == "partial"
    assert record["coverage"]["complete"] is False
    assert record["coverage"]["undelivered_work"]
    assert all("hbm" in str(item["resource"])
               for item in record["coverage"]["undelivered_work"])
    assert record["diagnostics"]["rate_comparison"] == []
    handoff = resource_phase_bindings(
        [affine_rule()], resource_services=[service_record],
        links={"hbm-copy-response": binding})
    assert handoff["bindings"] == []

    restored = binding_for(KernelProgram((
        _owned_physical_stage(placement={"hbm": ("r0",)}),)))
    record = assess_one(affine_rule(), [service_record], restored)
    assert record["status"] == "resource_phase_ready"
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["actual_work"] == pytest.approx(1536)
    expected_duration = 34 if work_scope == "per_owner" else 18
    assert 2 + rate["actual_work"] * 1e6 / rate["rate_used"] == (
        pytest.approx(expected_duration))


def _two_stage_work(mode):
    hbm = {"service_id": "hbm-copy-service", "stage": "hbm",
           "resource": "hbm", "work_unit": "bytes", "multiplicity": 1,
           "phase": "copy"}
    link = {"service_id": "link-copy-service", "stage": "link",
            "resource": "link", "work_unit": "bytes", "multiplicity": 1,
            "phase": "copy"}
    if mode in {"all", "partial"}:
        hbm = {**hbm, "term_index": 0}
    if mode == "all":
        link = {**link, "term_index": 1}
    return [hbm, link]


def _two_stage_binding(mode):
    return plan(
        program=two_stage_program(), runtime=two_stage_context(),
        fixed=[{"service_id": "hbm-copy-service", "stage": "hbm"},
               {"service_id": "link-copy-service", "stage": "link"}],
        work=_two_stage_work(mode))


def test_actual_coverage_is_independent_of_old_term_annotations():
    """All, partial and zero old term links give the same actual coverage."""
    expected_instances = [("hbm", 1536.0), ("link", 768.0)]
    for mode in ("all", "partial", "zero"):
        record = assess_one(two_term_rule(), [service(), link_service()],
                            _two_stage_binding(mode))
        assert record["status"] == "resource_phase_ready", mode
        coverage = record["coverage"]
        assert coverage["complete"] is True
        assert [(item["stage"], item["amount"])
                for item in coverage["actual_work_instances"]] == expected_instances
        rates = record["diagnostics"]["rate_comparison"]
        assert [(rate["stage"], rate["actual_work"], rate["charged_once"])
                for rate in rates] == [("hbm", 1536.0, True),
                                       ("link", 768.0, True)]
        assert coverage["fixed_total_us"] == pytest.approx(4)
        assert record["diagnostics"]["fixed_occurrence_count"] == 2
        handoff = resource_phase_bindings(
            [two_term_rule()], resource_services=[service(), link_service()],
            links={"hbm-copy-response": _two_stage_binding(mode)})
        assert len(handoff["bindings"]) == 1

    for mode in ("all", "partial", "zero"):
        binding = _two_stage_binding(mode)
        binding["work"] = [item for item in binding["work"]
                           if item["service_id"] != "link-copy-service"]
        record = assess_one(two_term_rule(), [service(), link_service()], binding)
        assert record["status"] == "partial", mode
        assert record["coverage"]["complete"] is False
        assert all(item["stage"] != "link"
                   for item in record["coverage"]["actual_work_instances"]
                   if item in record["coverage"]["undelivered_work"])
        assert any("actual_work_uncovered:link" in item
                   for item in record["blockers"]), mode
        handoff = resource_phase_bindings(
            [two_term_rule()], resource_services=[service(), link_service()],
            links={"hbm-copy-response": binding})
        assert handoff["bindings"] == []


def _physical_stage_and_context():
    physical = "hbm@r0:s0"
    fragment = ProgramFragment(
        name="hbm-read",
        work=WorkSet((WorkItem(resource=physical, amount=1536,
                               unit="bytes", source="actual-lowering"),)),
        repeats=1)
    physical_stage = Stage("hbm", work={physical: 1536},
                           work_units={physical: "bytes"}, fragment=fragment)
    runtime = RecipeContext(
        op=None, work={}, resource_names={"hbm": physical},
        rates={physical: 96000.0}, core_count=32,
        native_work={"hbm": 1536}, work_units={"hbm": "bytes"},
        resource_units={"hbm": 1}, service_identity={})
    return physical_stage, runtime, physical


def test_physical_binding_resource_is_equivalent_to_logical_role():
    physical_stage, runtime, physical = _physical_stage_and_context()
    logical = assess_one(
        affine_rule(), [service()],
        plan(program=KernelProgram((physical_stage,)), runtime=runtime))
    explicit = assess_one(
        affine_rule(), [service()],
        plan(program=KernelProgram((physical_stage,)), runtime=runtime,
             work=[{"term_index": 0, "service_id": "hbm-copy-service",
                    "stage": "hbm", "resource": physical,
                    "work_unit": "bytes", "multiplicity": 1,
                    "phase": "copy"}]))
    assert logical["status"] == explicit["status"] == "resource_phase_ready"
    for record in (logical, explicit):
        rate = record["diagnostics"]["rate_comparison"][0]
        assert rate["actual_work"] == pytest.approx(1536)
        assert rate["instance"] == ["hbm", physical]
        assert rate["charged_once"] is True
        assert record["coverage"]["fixed_total_us"] == pytest.approx(2)
        assert record["coverage"]["fixed"] == [{
            "service_id": "hbm-copy-service", "stage": "hbm",
            "phase": "copy", "new_fixed_us": 2.0, "native_boundary": ""}]
    assert service().duration_us(1536) == pytest.approx(18)


def _real_lowering(service_record, amount, *, repeats=1):
    runtime = hbm_context(native=amount * repeats)
    runtime = replace(
        runtime,
        service_identity={"hardware": {"key": "Ascend950DT_9582"}},
        service_boundary="resource_phase")
    runtime = replace(
        runtime, work=runtime.bind(Work(resources={"hbm": amount * repeats})))
    lowered = ProfileServiceRecipe(service_record).lower(runtime)
    physical = "hbm@r0:s0"
    fragment = ProgramFragment(
        name="hbm-copy",
        work=WorkSet((WorkItem(resource="hbm", amount=amount,
                               unit="bytes", source="t06-service-lowering"),)),
        repeats=repeats)
    service_stage = replace(lowered.stages[1], fragment=fragment,
                            work_units={physical: "bytes"})
    program = KernelProgram((lowered.stages[0], service_stage))
    return program, runtime, lowered, physical


@pytest.mark.parametrize("rate, expected", (
    (96e6, {1536: 18, 2304: 26}),
    (128e6, {1536: 14, 2304: 20}),
))
def test_real_lowering_consumes_h5_formula_and_inclusive_fixed_once(rate, expected):
    for amount, duration in expected.items():
        service_record = service(
            supply={"kind": "absolute_rate", "rate": rate})
        program, runtime, lowered, physical = _real_lowering(
            service_record, amount)
        assert program.stages[0].name == "setup"
        assert program.stages[0].latency == pytest.approx(
            service_record.startup_us / 1000.0)
        assert dict(program.stages[1].work) == {physical: amount}
        runtime = replace(runtime, invocation_timing=InvocationTiming(
            startup_ms=service_record.startup_us / 1000.0))
        binding = {
            "replacement_kind": "independent_service", "scope": "complete_rule",
            "program": program, "context": runtime,
            "condition_context": {"hardware": "Ascend950DT_9582",
                                  "recipe": "copy"},
            "fixed": [{"service_id": "hbm-copy-service",
                       "stage": "setup" if amount == 2304 else "service"}],
            "work": [{"term_index": 0, "service_id": "hbm-copy-service",
                      "stage": "service", "resource": "hbm",
                      "work_unit": "bytes", "multiplicity": 1,
                      "phase": "copy"}]}
        record = assess_one(affine_rule(), [service_record], binding)
        assert record["status"] == "resource_phase_ready"
        coverage = record["coverage"]
        assert coverage["fixed_total_us"] == pytest.approx(2)
        assert record["diagnostics"]["fixed_occurrence_count"] == 1
        assert coverage["invocation"]["service_fixed_covers_startup"] is True
        assert [item["actual_stage"]
                for item in coverage["fixed_occurrences"]] == ["setup"]
        rate_row = record["diagnostics"]["rate_comparison"][0]
        assert rate_row["actual_work"] == pytest.approx(amount)
        assert rate_row["actual_unit"] == "bytes"
        assert rate_row["charged_once"] is True
        assert service_record.duration_us(amount) == pytest.approx(duration)


def test_repeat_work_shares_one_inclusive_invocation_fixed():
    service_record = service()
    program, runtime, lowered, physical = _real_lowering(
        service_record, 1536, repeats=3)
    runtime = replace(runtime, invocation_timing=InvocationTiming(
        startup_ms=service_record.startup_us / 1000.0))
    binding = {
        "replacement_kind": "independent_service", "scope": "complete_rule",
        "program": program, "context": runtime,
        "condition_context": {"hardware": "Ascend950DT_9582", "recipe": "copy"},
        "fixed": [{"service_id": "hbm-copy-service", "stage": "service"}],
        "work": [{"term_index": 0, "service_id": "hbm-copy-service",
                  "stage": "service", "resource": "hbm", "work_unit": "bytes",
                  "multiplicity": 3, "phase": "copy"}]}
    record = assess_one(affine_rule(), [service_record], binding)
    assert record["status"] == "resource_phase_ready"
    coverage = record["coverage"]
    rate = record["diagnostics"]["rate_comparison"][0]
    assert rate["actual_work"] == pytest.approx(4608)
    assert rate["actual_repeats"] == 3
    assert rate["charged_once"] is True
    assert coverage["fixed_total_us"] == pytest.approx(2)
    assert record["diagnostics"]["fixed_occurrence_count"] == 1
    assert coverage["invocation"]["service_fixed_covers_startup"] is True
    assert service_record.duration_us(4608) == pytest.approx(50)


def test_separate_invocation_startup_is_charged_once_not_twice():
    zero_fixed = service(startup_us=0,
                         supply={"kind": "absolute_rate", "rate": 96e6})
    assert zero_fixed.fixed_us == 0
    runtime = replace(hbm_context(),
                      invocation_timing=InvocationTiming(startup_ms=0.002))
    record = assess_one(affine_rule(), [zero_fixed], plan(runtime=runtime))
    assert record["status"] == "resource_phase_ready"
    coverage = record["coverage"]
    assert coverage["fixed_total_us"] == pytest.approx(2)
    assert record["diagnostics"]["fixed_occurrence_count"] == 1
    assert coverage["fixed_occurrences"] == [{
        "kind": "invocation_startup", "service_id": None,
        "stage": "hbm", "actual_stage": "hbm", "phase": None,
        "fixed_us": 2.0, "native_boundary": "",
        "included_fixed_effects": (
            "T04 actual invocation startup declared separately from the "
            "selected service formula and charged once"),
        "fixed_components": "invocation_startup",
        "covers_invocation_startup": True,
        "invocation_startup_us": 2.0}]
    assert coverage["invocation"]["service_fixed_covers_startup"] is False


def test_two_ready_copies_share_96b_per_us_and_independent_copy_takes_16us():
    raw = hardware_case()
    raw["machine"]["device"]["rates"] = {"memory": 96e6, "vector": 1e6}
    case = parse_case(raw)
    model = ServiceCurve(
        resource="hbm", work_unit="bytes",
        identity=service_identity(case, copy_op(1536)),
        boundary="device_kernel", active_units=4, startup_ms=0,
        knots=((1536, 0.016), (3072, 0.032)))
    left = copy_op(1536, stream="left")
    right = copy_op(1536, op_id="right", stream="right")
    step, execution = execute((left, right), ResourceService(model), raw=raw)
    assert step.completion.time == pytest.approx(0.032)
    assert execution.engine.serviced["memory@r0:s0"] == pytest.approx(3072)
    assert execution.cost.summary()["service_models"][0]["queries"] == {
        "anchor": 2}
    single, single_execution = execute((left,), ResourceService(model), raw=raw)
    assert single.completion.time == pytest.approx(0.016)
    assert single_execution.engine.serviced["memory@r0:s0"] == pytest.approx(
        1536)


def test_t06_fixed_startup_runs_once_on_the_ordinary_engine_path():
    raw = hardware_case()
    raw["machine"]["device"]["rates"] = {"memory": 96e6, "vector": 1e6}
    service_record = normalize_resource_service({
        "id": "hbm-copy-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
        "boundary": "device_kernel", "startup_us": 2, "response_us": 0,
        "supply": {"kind": "absolute_rate", "rate": 96e6},
        "rate_scope": "aggregate", "active_units": 4, "conditions": {},
        "identification": {
            "status": "identified", "resource_identified": True,
            "rate_source": "fitted", "fixed_source": "fitted",
            "fixed_identified": True, "train_work": [1536.0, 3072.0]},
        "provenance": {"kind": "calibrated_resource_service",
                       "training_sources": ["new-run-a", "new-run-b"]},
    })
    step, execution = execute(
        (copy_op(1536),), ProfileServiceRecipe(service_record), raw=raw)
    assert step.completion.time == pytest.approx(0.018)
    assert execution.engine.serviced["memory@r0:s0"] == pytest.approx(1536)
    boundaries = [(event["task"], event["start_ms"], event["end_ms"])
                  for event in execution.engine.trace
                  if event["task"].endswith(":setup")
                  or event["task"].endswith(":service")]
    assert [event[0] for event in boundaries] == [
        "r0:i0:copy:setup", "r0:i0:copy:service"]
    for _task, start, end in boundaries:
        assert start == pytest.approx(0.0 if _task.endswith(":setup") else 0.002)
        assert end == pytest.approx(0.002 if _task.endswith(":setup") else 0.018)


def test_response_only_fixed_and_invocation_startup_are_distinct_effects():
    response_only = service(startup_us=1,
                            supply={"kind": "absolute_rate", "rate": 96e6})
    runtime = replace(hbm_context(),
                      invocation_timing=InvocationTiming(startup_ms=0.002))
    record = assess_one(affine_rule(), [response_only], plan(runtime=runtime))
    assert record["status"] == "resource_phase_ready"
    coverage = record["coverage"]
    assert coverage["fixed_total_us"] == pytest.approx(3)
    assert record["diagnostics"]["fixed_occurrence_count"] == 2
    assert [(item["kind"], item["fixed_us"])
            for item in coverage["fixed_occurrences"]] == [
                ("service_fixed", 1.0), ("invocation_startup", 2.0)]
