"""A5 resource/phase calibration: one rate, one fixed term, no joint overclaim."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.observations.calibration import (
    apply_resource_profile,
    export_resource_profile,
    load_resource_profile,
    resource_service_binding,
    service_conditions_match,
)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA,
    RESOURCE_SERVICE_SCHEMA,
    normalize_resource_service,
)
from mindie_sim.observations.service_fitting import (
    bind_capture_native_plan, build_actual_evidence, capture_image_identity,
    capture_measurement,
    calibration_role_partitions, fit_actual_training_evidence,
    fit_resource_service, fit_resource_service_from_role_records,
    join_capture_measurement_and_plan, load_native_cohort_validation,
    load_role_ledger, logical_capture_work, read_capture_locators,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


ROOT = Path(__file__).parents[1]
LOCAL_SERVICES = ROOT / "data/observations/a5-local-service-regimes-20261030/parameters.json"
JOINT_COPY = ROOT / "data/observations/a5-copy-joint-service-20261030/parameters.json"
NOMINAL_BPS = {"mte2_aiv": 4224000000000.0, "mte3": 13516800000000.0}
NOMINAL = {
    name: {"rate": rate, "work_unit": "bytes", "rate_scope": "aggregate"}
    for name, rate in NOMINAL_BPS.items()
}
BYTES_PER_ROW = 6 * 3072 * 2
ANCHOR_CAPTURE = Path(
    "/Volumes/data/mindie-sim-data/a5/"
    "deepseek_v4_1_flash_fp8_high_conc_decode_tpot30ms_c64_20261030/"
    "kernel_details.csv")
NATIVE_VALIDATION = Path(
    "/Volumes/data/Codex/2026-11-12/kan-x/work/a5-implementation/jobs/"
    "A5-T03/out/native/native_cohort_validation.csv")
ANCHOR_COHORT = ("sha256:b007c5ec0d17e5160f1a0a284c12d25f2d4001408092bb3a"
                 "550c5c3e3732416a")


def service(**changes):
    record = {
        "id": "a5-service",
        "scope": "independent_resource",
        "resources": ["mte2_aiv"],
        "phase": "prefill",
        "work_unit": "bytes",
        "boundary": "resource_phase",
        "startup_us": 0,
        "response_us": 0,
        "active_units": 1,
        "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "identification": {"resource_identified": True},
    }
    record.update(changes)
    return record


def profile(*services, **changes):
    result = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "Ascend950DT_9582",
        "services": list(services),
    }
    result.update(changes)
    return result


def a5_gather_packet():
    rules = json.loads(LOCAL_SERVICES.read_text())["rules"]
    rule = next(item for item in rules if item["id"] == "a5-strided-state-gather")
    fixed_us = rule["response"]["fixed_us"]
    us_per_row = rule["response"]["terms"][0]["us_per_unit"]
    effective_rate = BYTES_PER_ROW / us_per_row * 1e6
    return {
        "id": "a5-strided-state-gather-calibrated",
        "scope": "joint_phase",
        "resources": ["mte2_aiv"],
        "phase": "prefill",
        "work_unit": "bytes",
        "boundary": "device_kernel",
        "active_units": 64,
        "conditions": {"hardware": "Ascend950DT_9582", "dtype": "bf16"},
        "provenance": {
            "source": str(LOCAL_SERVICES.relative_to(ROOT)),
            "observation_kind": "native_call",
            "window": "joint",
            "rate_interpretation": rule["provenance"]["rate_interpretation"],
        },
        "nominal_rates": {"mte2_aiv": NOMINAL["mte2_aiv"]},
        "samples": [
            {"work": 16 * BYTES_PER_ROW, "duration_us": 9.384521568627452,
             "run_id": "b16", "repeat_index": 0, "split": "train",
             "source": "a5#b16"},
            {"work": 64 * BYTES_PER_ROW, "duration_us": 26.798966793168873,
             "run_id": "b64", "repeat_index": 0, "split": "train",
             "source": "a5#b64"},
            {"work": 32 * BYTES_PER_ROW, "duration_us": 15.043888994307407,
             "run_id": "b32", "repeat_index": 0, "split": "heldout",
             "source": "a5#b32"},
        ],
        "fixed_reference_us": fixed_us,
        "rate_reference": effective_rate,
    }


def test_multiplier_is_resolved_once_and_reload_is_idempotent():
    model = normalize_resource_service(service(
        startup_us=2,
        response_us=1,
        active_units=4,
        supply={"kind": "nominal_multiplier", "nominal_rate": 1_000_000,
                "multiplier": 0.25},
    ))
    assert model.effective_rate == pytest.approx(250_000)
    assert model.fixed_us == pytest.approx(3)
    assert model.duration_us(1000) == pytest.approx(3 + 1000 * 1e6 / 250_000)
    exported = model.to_dict()
    assert exported["schema"] == RESOURCE_SERVICE_SCHEMA
    assert exported["normalization"]["input_kind"] == "nominal_multiplier"
    reloaded = normalize_resource_service(exported)
    assert reloaded.effective_rate == pytest.approx(model.effective_rate)
    assert reloaded.fixed_us == pytest.approx(model.fixed_us)
    assert reloaded.duration_us(1000) == pytest.approx(model.duration_us(1000))
    assert reloaded.model_id == model.model_id
    assert reloaded.normalization["reused"] is True


def test_absolute_rate_and_multiplier_share_the_same_normalized_identity():
    absolute = normalize_resource_service(service(
        supply={"kind": "absolute_rate", "rate": 250_000,
                "nominal_rate": 1_000_000},
    ))
    multiplier = normalize_resource_service(service(
        supply={"kind": "nominal_multiplier", "nominal_rate": 1_000_000,
                "multiplier": 0.25},
    ))
    assert absolute.effective_rate == multiplier.effective_rate == 250_000
    assert absolute.nominal_multiplier == multiplier.nominal_multiplier == 0.25
    assert absolute.model_id == multiplier.model_id
    assert absolute.duration_us(4000) == multiplier.duration_us(4000)


def test_ambiguous_incomplete_or_wrong_unit_supply_is_rejected():
    with pytest.raises(ValueError, match="supply_representation_ambiguous"):
        normalize_resource_service(service(
            supply={"kind": "absolute_rate", "rate": 1000,
                    "nominal_rate": 1000, "multiplier": 1},
        ))
    with pytest.raises(ValueError, match="multiplier_requires_nominal_rate"):
        normalize_resource_service(service(
            supply={"kind": "nominal_multiplier", "multiplier": 0.5},
        ))
    with pytest.raises(ValueError, match="nominal_rate_unit_mismatch"):
        normalize_resource_service(service(
            supply={"kind": "nominal_multiplier", "nominal_rate": 1000,
                    "multiplier": 0.5, "nominal_unit": "rows"},
        ))
    with pytest.raises(ValueError, match="requires_nonempty_scope"):
        normalize_resource_service({key: value for key, value in service().items()
                                    if key != "scope"})
    with pytest.raises(ValueError, match="supply_unknown_fields"):
        normalize_resource_service(service(
            supply={"kind": "absolute_rate", "rate": 1000, "nominal": 1000}))
    with pytest.raises(ValueError, match="identified_status_requires"):
        normalize_resource_service(service(
            identification={"status": "identified", "resource_identified": False}))
    with pytest.raises(ValueError, match="parameter_class_conflict"):
        normalize_resource_service(service(
            provenance={"parameter_class": "nominal_specification"}))
    with pytest.raises(ValueError, match="nominal_rate_cannot_be_identified"):
        normalize_resource_service(service(
            supply={"kind": "nominal", "nominal_rate": 1000},
            identification={"status": "identified", "resource_identified": True}))
    with pytest.raises(ValueError, match="nominal_rate_scope_mismatch"):
        load_resource_profile(
            profile(service(
                rate_scope="per_owner",
                supply={"kind": "nominal_multiplier", "multiplier": 0.5})),
            nominal_rates={"mte2_aiv": {"rate": 1_000_000, "work_unit": "bytes",
                                        "rate_scope": "aggregate"}})
    with pytest.raises(ValueError, match="bare_nominal_rate_requires_explicit"):
        load_resource_profile(
            profile(service(
                rate_scope="per_owner",
                supply={"kind": "nominal_multiplier", "multiplier": 0.5})),
            nominal_rates={"mte2_aiv": 1_000_000})
    with pytest.raises(ValueError, match="requires_explicit_work_unit"):
        load_resource_profile(
            profile(service(
                supply={"kind": "nominal_multiplier", "multiplier": 0.5})),
            nominal_rates={"mte2_aiv": {"rate": 1_000_000,
                                        "rate_scope": "aggregate"}})
    with pytest.raises(ValueError, match="requires_explicit_rate_scope"):
        load_resource_profile(
            profile(service(
                supply={"kind": "nominal_multiplier", "multiplier": 0.5})),
            nominal_rates={"mte2_aiv": {"rate": 1_000_000,
                                        "work_unit": "bytes"}})


def test_per_owner_rate_expands_once_and_keeps_both_scopes_visible():
    model = normalize_resource_service(service(
        rate_scope="per_owner",
        active_units=4,
        supply={"kind": "nominal_multiplier", "nominal_rate": 1_000_000,
                "multiplier": 0.5},
    ))
    assert model.effective_rate == 500_000            # declared per owner
    assert model.per_owner_rate == 500_000
    assert model.aggregate_rate == 2_000_000
    assert model.duration_us(2_000_000, work_scope="aggregate") == pytest.approx(1_000_000)
    assert model.duration_us(500_000, work_scope="per_owner") == pytest.approx(1_000_000)
    reloaded = normalize_resource_service(model.to_dict())
    assert reloaded.aggregate_rate == model.aggregate_rate
    assert reloaded.per_owner_rate == model.per_owner_rate
    assert reloaded.duration_us(2_000_000, work_scope="aggregate") == pytest.approx(1_000_000)


def test_joint_phase_predicts_its_window_but_not_internal_resources():
    joint = normalize_resource_service(service(
        scope="joint_phase",
        resources=["mte2_aiv", "mte3"],
        startup_us=3,
        supply={"kind": "absolute_rate", "rate": 1000},
        identification={"status": "joint_identified",
                        "resource_identified": False},
    ))
    assert joint.kind == "joint_phase"
    assert joint.resource is None
    assert joint.duration_us(2000) == pytest.approx(2_000_003)
    assert joint.prediction(2000)["basis"] == "formula"
    assert "resource_rates" not in joint.to_dict()
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        joint.counterfactual(2000, nominal_rate=2000)
    with pytest.raises(ValueError, match="joint_service_cannot_identify"):
        normalize_resource_service(service(
            scope="joint_phase",
            resources=["mte2_aiv", "mte3"],
            identification={"resource_identified": True},
        ))
    with pytest.raises(ValueError, match="requires_exactly_one_resource"):
        normalize_resource_service(service(
            scope="independent_resource",
            resources=["mte2_aiv", "mte3"],
        ))


def test_independent_counterfactual_scales_rate_once_and_keeps_fixed_term():
    model = normalize_resource_service(service(
        startup_us=2,
        response_us=1,
        supply={"kind": "nominal_multiplier", "nominal_rate": 1_000_000,
                "multiplier": 0.5},
    ))
    changed = model.counterfactual(1000, nominal_rate=2_000_000)
    assert changed["effective_rate"] == pytest.approx(1_000_000)
    assert changed["fixed_us"] == pytest.approx(3)
    assert changed["duration_us"] == pytest.approx(3 + 1000 * 1e6 / 1_000_000)
    assert changed["basis"] == "declared_nominal_multiplier"
    with pytest.raises(ValueError, match="counterfactual_requires_nominal_rate"):
        normalize_resource_service(service(
            supply={"kind": "absolute_rate", "rate": 1000},
        )).counterfactual(10, nominal_rate=100)


def test_a5_nominal_multiplier_profile_uses_arch3510_rate_once():
    loaded = load_resource_profile(
        profile(service(
            id="a5-mte2-effective",
            supply={"kind": "nominal_multiplier",
                    "nominal_rate": NOMINAL_BPS["mte2_aiv"],
                    "multiplier": 0.25},
        )),
        source_kind="builtin",
    )
    recorded = loaded["services"][0]
    assert recorded["effective_rate"] == pytest.approx(NOMINAL_BPS["mte2_aiv"] * 0.25)
    assert recorded["nominal_multiplier"] == pytest.approx(0.25)
    assert recorded["normalization"]["input_kind"] == "nominal_multiplier"
    assert recorded["normalization"]["effective_rate"] == pytest.approx(
        NOMINAL_BPS["mte2_aiv"] * 0.25)
    assert recorded["provenance"]["parameter_class"] == "effective_resource_service"


def test_conditions_match_only_the_dependencies_the_stage_declares():
    model = normalize_resource_service(service(
        conditions={
            "hardware": "Ascend950DT_9582",
            "dtype": "bf16",
            "resource_units.mte2_aiv": 64,
            "rates.mte2_aiv": NOMINAL_BPS["mte2_aiv"],
            "modes.sparse": True,
        },
    ))
    broader_contract = {
        "hardware": "Ascend950DT_9582",
        "dtype": "bf16",
        "resource_units": {"mte2_aiv": 64, "mte3": 64, "vector": 64},
        "rates": {"mte2_aiv": NOMINAL_BPS["mte2_aiv"],
                  "mte3": NOMINAL_BPS["mte3"]},
        "modes": {"sparse": True, "dense": True},
        "environment": {"cann": "8.5", "driver": "unrelated"},
        "new_physical_contract": {"extra": "ignored"},
    }
    matched = service_conditions_match(model, broader_contract)
    assert matched["applies"] is True
    assert len(matched["checks"]) == 5
    assert "unrelated" in matched["policy"]
    changed_rate = deepcopy(broader_contract)
    changed_rate["rates"]["mte2_aiv"] *= 0.9
    assert service_conditions_match(model, changed_rate)["applies"] is False
    changed_units = deepcopy(broader_contract)
    changed_units["resource_units"]["mte2_aiv"] = 32
    assert service_conditions_match(model, changed_units)["applies"] is False
    changed_mode = deepcopy(broader_contract)
    changed_mode["modes"]["sparse"] = False
    assert service_conditions_match(model, changed_mode)["applies"] is False
    missing = deepcopy(broader_contract)
    del missing["rates"]["mte2_aiv"]
    assert service_conditions_match(model, missing)["applies"] is False
    # Normalized dict exports carry the same declared conditions.
    assert service_conditions_match(model.to_dict(), broader_contract)["applies"] is True
    assert service_conditions_match(
        normalize_resource_service(service()), broader_contract)["applies"] is True


def test_nominal_rates_require_actual_unit_and_explicit_conversion():
    rows_service = service(
        id="rows-service", work_unit="rows",
        supply={"kind": "nominal_multiplier", "multiplier": 1.0})
    converted = load_resource_profile(
        profile(rows_service),
        nominal_rates={"mte2_aiv": {
            "rate": NOMINAL_BPS["mte2_aiv"], "work_unit": "bytes",
            "rate_scope": "aggregate",
            "conversion": {"target_work_unit": "rows",
                           "source_units_per_target_unit": BYTES_PER_ROW},
        }})
    recorded = converted["services"][0]
    assert recorded["effective_rate"] == pytest.approx(
        NOMINAL_BPS["mte2_aiv"] / BYTES_PER_ROW)
    assert recorded["rate_representation"]["multiplier_source"] == "declared"
    with pytest.raises(ValueError, match="nominal_rate_unit_mismatch"):
        load_resource_profile(
            profile(rows_service),
            nominal_rates={"mte2_aiv": {
                "rate": NOMINAL_BPS["mte2_aiv"], "work_unit": "bytes",
                "rate_scope": "aggregate"}})
    with pytest.raises(ValueError, match="nominal_rate_unit_mismatch"):
        load_resource_profile(
            profile(rows_service),
            nominal_rates={"mte2_aiv": {
                "rate": NOMINAL_BPS["mte2_aiv"], "work_unit": "bytes",
                "rate_scope": "aggregate",
                "conversion": {"target_work_unit": "tokens",
                               "source_units_per_target_unit": BYTES_PER_ROW}}})
    with pytest.raises(ValueError, match="conversion_not_needed"):
        load_resource_profile(
            profile(service(id="bytes-service")),
            nominal_rates={"mte2_aiv": {
                "rate": NOMINAL_BPS["mte2_aiv"], "work_unit": "bytes",
                "rate_scope": "aggregate",
                "conversion": {"target_work_unit": "bytes",
                               "source_units_per_target_unit": 1}}})


def test_mapping_and_file_profiles_share_hardware_validation_and_source(tmp_path):
    inconsistent = profile(service(
        id="cross-profile", conditions={"hardware": "Ascend950PR_9579"}))
    case = {"runtime": {"metadata": {"keep": 1}}, "model": {"name": "m"}}
    original = deepcopy(case)
    with pytest.raises(ValueError, match="hardware_mismatch"):
        load_resource_profile(inconsistent)
    with pytest.raises(ValueError, match="hardware_mismatch"):
        apply_resource_profile(case, inconsistent)
    inconsistent_path = tmp_path / "cross-profile.json"
    inconsistent_path.write_text(json.dumps(inconsistent))
    with pytest.raises(ValueError, match="hardware_mismatch"):
        load_resource_profile(inconsistent_path)
    with pytest.raises(ValueError, match="hardware_mismatch"):
        apply_resource_profile(case, inconsistent_path)
    assert case == original

    valid_path = tmp_path / "valid-profile.json"
    valid_path.write_text(json.dumps(profile(service(id="valid-profile"))))
    loaded = load_resource_profile(valid_path)
    reloaded = load_resource_profile(loaded)
    assert reloaded["source"] == loaded["source"]
    assert loaded["source"]["kind"] == "file"
    assert loaded["source"]["sha256"]
    applied = apply_resource_profile(case, loaded)
    assert case == original
    assert (applied["runtime"]["metadata"]["resource_service_profile_source"]
            == loaded["source"])
    assert applied["runtime"]["metadata"]["keep"] == 1


def test_h5_pure_rate_intervention_and_multiplier_source_labels():
    absolute = normalize_resource_service(service(
        id="h5-absolute", resources=["hbm"], startup_us=2,
        supply={"kind": "absolute_rate", "rate": 96_000_000,
                "nominal_rate": 128_000_000}))
    assert absolute.duration_us(1536) == pytest.approx(18)
    scaled = absolute.counterfactual(1536, nominal_rate=128_000_000)
    assert scaled["effective_rate"] == pytest.approx(96_000_000)
    assert scaled["duration_us"] == pytest.approx(18)
    assert scaled["basis"] == "effective_rate_ratio_to_declared_nominal"
    assert (scaled["multiplier_source"]
            == "inferred_from_effective_rate_and_nominal")
    assert (absolute.normalization["multiplier_source"]
            == "inferred_from_effective_rate_and_nominal")
    pure = absolute.counterfactual_effective_rate(1536, effective_rate=128_000_000)
    assert pure["effective_rate"] == pytest.approx(128_000_000)
    assert pure["duration_us"] == pytest.approx(14)
    assert pure["basis"] == "explicit_effective_rate_intervention"
    reloaded = normalize_resource_service(absolute.to_dict())
    assert (reloaded.rate_representation["multiplier_source"]
            == "inferred_from_effective_rate_and_nominal")

    declared = normalize_resource_service(service(
        id="h5-declared-multiplier", resources=["hbm"], startup_us=2,
        supply={"kind": "nominal_multiplier", "nominal_rate": 128_000_000,
                "multiplier": 0.75}))
    assert declared.rate_representation["multiplier_source"] == "declared"
    assert declared.normalization["multiplier_source"] == "declared"
    declared_scaled = declared.counterfactual(1536, nominal_rate=128_000_000)
    assert declared_scaled["effective_rate"] == pytest.approx(96_000_000)
    assert declared_scaled["duration_us"] == pytest.approx(18)
    assert declared_scaled["basis"] == "declared_nominal_multiplier"

    nominal = normalize_resource_service(service(
        id="h5-nominal", resources=["hbm"], startup_us=2,
        supply={"kind": "nominal", "nominal_rate": 96_000_000}))
    assert nominal.rate_representation["multiplier_source"] == "identity"
    assert nominal.normalization["multiplier_source"] == "identity"
    nominal_scaled = nominal.counterfactual(1536, nominal_rate=128_000_000)
    assert nominal_scaled["effective_rate"] == pytest.approx(128_000_000)
    assert nominal_scaled["duration_us"] == pytest.approx(14)
    assert nominal_scaled["basis"] == "declared_nominal_rate_ratio"


def test_profile_loader_uses_one_path_for_builtin_and_user_file(tmp_path):
    builtin = load_resource_profile(
        profile(service(id="a5-builtin")), source_kind="builtin")
    user_file = tmp_path / "a5-user-profile.json"
    user_file.write_text(json.dumps(profile(service(id="a5-builtin"))))
    user = load_resource_profile(user_file, hardware="Ascend950DT_9582")
    assert builtin["service_ids"] == user["service_ids"] == ["a5-builtin"]
    assert builtin["services"][0]["model_id"] == user["services"][0]["model_id"]
    assert builtin["source"]["kind"] == "builtin"
    assert user["source"]["kind"] == "file"
    assert user["source"]["sha256"]
    assert user["services"][0]["conditions"]["hardware"] == "Ascend950DT_9582"
    assert user["services"][0]["identification"]["status"] == "identified"
    assert (user["services"][0]["provenance"]["parameter_class"]
            == "effective_resource_service")
    out = tmp_path / "a5-export.json"
    written = export_resource_profile(user, path=out)
    assert written["bytes"] == out.stat().st_size
    again = load_resource_profile(out)
    assert again["services"][0]["model_id"] == user["services"][0]["model_id"]


def test_profile_loader_fails_explicitly_instead_of_falling_back_to_nominal(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="source_missing"):
        load_resource_profile(missing)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "unknown", "services": []}))
    with pytest.raises(ValueError, match="schema_unknown"):
        load_resource_profile(bad)
    with pytest.raises(ValueError, match="multiplier_requires_nominal_rate"):
        load_resource_profile(profile(service(
            supply={"kind": "nominal_multiplier", "multiplier": 0.5})))
    with pytest.raises(ValueError, match="hardware_mismatch"):
        load_resource_profile(profile(service()), hardware="Ascend910B2")
    normalized = normalize_resource_service(
        service(conditions={"hardware": "Ascend950DT_9582"})).to_dict()
    with pytest.raises(ValueError, match="hardware_mismatch"):
        load_resource_profile(
            {"schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
             "services": [normalized]},
            hardware="Ascend910B2")


def test_apply_resource_profile_is_an_additive_metadata_handoff():
    case = {"runtime": {"metadata": {"existing": 1}}, "model": {"name": "m"}}
    original = deepcopy(case)
    loaded = load_resource_profile(profile(service()))
    result = apply_resource_profile(case, loaded)
    assert case == original
    metadata = result["runtime"]["metadata"]
    assert metadata["existing"] == 1
    assert metadata["resource_service_profile"]["service_ids"] == ["a5-service"]
    assert metadata["resource_service_profile_source"]["kind"] == "in_memory"
    assert "kernels" not in result["runtime"]


def test_a5_strided_gather_fit_is_joint_and_predicts_unseen_rows():
    packet = a5_gather_packet()
    fitted = fit_resource_service(packet, nominal_rates=packet["nominal_rates"])
    assert fitted["service"]["kind"] == "joint_phase"
    assert fitted["service"]["identification"]["resource_identified"] is False
    assert (fitted["service"]["provenance"]["rate_interpretation"]
            == packet["provenance"]["rate_interpretation"])
    assert (fitted["service"]["provenance"]["normalized_rate_interpretation"]
            == "joint_phase_service_rate_not_an_isolated_internal_resource_rate")
    assert fitted["fit"]["fixed_us"] == pytest.approx(
        packet["fixed_reference_us"], rel=1e-9)
    assert fitted["fit"]["effective_rate"] == pytest.approx(
        packet["rate_reference"], rel=1e-9)
    assert fitted["fit"]["nominal_multiplier"] == pytest.approx(
        packet["rate_reference"] / NOMINAL_BPS["mte2_aiv"], rel=1e-9)
    assert len(fitted["heldout"]) == 1
    assert abs(fitted["heldout"][0]["relative_error"]) < 0.02
    unseen = 128 * BYTES_PER_ROW
    prediction = normalize_resource_service(fitted["service"]).prediction(unseen)
    assert prediction["basis"] == "formula"
    assert prediction["extrapolated"] is True
    assert prediction["duration_us"] == pytest.approx(
        packet["fixed_reference_us"] + 128 * (1e6 / (packet["rate_reference"] / BYTES_PER_ROW)))
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        normalize_resource_service(fitted["service"]).counterfactual(
            1, nominal_rate=1)

    changed = deepcopy(packet)
    changed["samples"][-1]["duration_us"] *= 30
    second = fit_resource_service(changed, nominal_rates=packet["nominal_rates"])
    assert second["service"] == fitted["service"]
    assert second["fit"] == fitted["fit"]


def test_fit_uses_equal_run_weights_and_keeps_heldout_out():
    samples = [
        {"work": 1000, "duration_us": 1, "run_id": "a", "repeat_index": index,
         "split": "train", "source": "a"}
        for index in range(1)
    ] + [
        {"work": 1000, "duration_us": 3, "run_id": "b", "repeat_index": index,
         "split": "train", "source": "b"}
        for index in range(19)
    ] + [
        {"work": 3000, "duration_us": 5, "run_id": "a", "repeat_index": 0,
         "split": "train", "source": "a"},
        {"work": 3000, "duration_us": 5, "run_id": "b", "repeat_index": 0,
         "split": "train", "source": "b"},
        {"work": 2000, "duration_us": 3.5, "run_id": "held", "repeat_index": 0,
         "split": "heldout", "source": "held"},
    ]
    packet = {
        "id": "equal-weight", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "samples": samples,
    }
    fitted = fit_resource_service(packet)
    # Run medians are 1 and 3 at 1000; their median is 2, not the 19-repeat 3.
    assert fitted["fit"]["fixed_us"] == pytest.approx(0.5)
    assert fitted["fit"]["effective_rate"] == pytest.approx(1e6 / 1.5e-3)
    assert fitted["heldout"][0]["predicted_us"] == pytest.approx(3.5)
    duplicated = deepcopy(packet)
    duplicated["samples"] += deepcopy(duplicated["samples"])
    assert fit_resource_service(duplicated)["service"] == fitted["service"]


def test_fit_rejects_missing_identification_axis_and_whole_step_prices():
    one_work = {
        "id": "one-work", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "samples": [{"work": 1000, "duration_us": 2, "run_id": "a",
                     "repeat_index": 0, "split": "train", "source": "a"}],
    }
    with pytest.raises(ValueError, match="two_work_sizes_or_a_declared_rate"):
        fit_resource_service(one_work)
    with pytest.raises(ValueError, match="boundary_must_be_resource_phase"):
        fit_resource_service({**one_work, "boundary": "step"})
    with pytest.raises(ValueError, match="rejects_target_duration"):
        fit_resource_service({**one_work, "target_ms": 2})
    with pytest.raises(ValueError, match="mixed_window_cannot_identify"):
        fit_resource_service({**one_work, "resources": ["hbm", "mte2_aiv"]})
    with pytest.raises(ValueError, match="mixed_window_cannot_identify"):
        fit_resource_service({**one_work,
                              "provenance": {"window": "joint"}})
    joint = {
        "id": "joint-step", "scope": "joint_phase",
        "resources": ["a", "b"], "phase": "primitive",
        "work_unit": "busy_ns", "boundary": "device_kernel",
        "provenance": {"window": "step"},
        "samples": [
            {"work": 10, "duration_us": 13, "run_id": "a", "repeat_index": 0,
             "split": "train", "source": "x"},
            {"work": 20, "duration_us": 23, "run_id": "b", "repeat_index": 0,
             "split": "train", "source": "x"},
        ],
    }
    with pytest.raises(ValueError, match="rejects_step_or_body_window"):
        fit_resource_service(joint)


def test_declared_rate_calibrates_fixed_from_one_small_packet():
    packet = {
        "id": "small-packet", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "response_us": 1,
        "supply": {"kind": "absolute_rate", "rate": 500_000_000},
        "samples": [{"work": 1000, "duration_us": 5, "run_id": "a",
                     "repeat_index": 0, "split": "train", "source": "a"}],
    }
    fitted = fit_resource_service(packet)
    assert fitted["fit"]["rate_source"] == "declared"
    assert fitted["fit"]["fixed_us"] == pytest.approx(3)
    assert fitted["fit"]["startup_us"] == pytest.approx(2)
    assert fitted["fit"]["response_us"] == pytest.approx(1)
    assert fitted["service"]["identification"]["status"] == "partially_identified"
    assert fitted["service"]["identification"]["resource_identified"] is False
    model = normalize_resource_service(fitted["service"])
    assert model.duration_us(2000) == pytest.approx(7)
    assert model.duration_us(500) == pytest.approx(4)  # new shape: 3 + 1

    contradicted = deepcopy(packet)
    contradicted["samples"][0]["duration_us"] = 1
    with pytest.raises(ValueError, match="contradict_declared_rate"):
        fit_resource_service(contradicted)


def test_declared_nominal_multiplier_fit_preserves_the_relative_representation():
    packet = {
        "id": "multiplier-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "supply": {"kind": "nominal_multiplier", "nominal_rate": 1_000_000_000,
                   "multiplier": 0.5},
        "samples": [{"work": 1000, "duration_us": 5, "run_id": "a",
                     "repeat_index": 0, "split": "train", "source": "a"}],
    }
    fitted = fit_resource_service(packet)
    service = fitted["service"]
    assert service["rate_representation"]["input_kind"] == "nominal_multiplier"
    assert service["nominal_multiplier"] == pytest.approx(0.5)
    assert service["effective_rate"] == pytest.approx(500_000_000)
    assert service["identification"]["status"] == "partially_identified"
    assert service["provenance"]["parameter_class"] == "effective_resource_service"
    assert normalize_resource_service(service).duration_us(2000) == pytest.approx(7)


def test_declared_nominal_supply_is_labeled_as_a_nominal_specification():
    packet = {
        "id": "nominal-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "supply": {"kind": "nominal", "nominal_rate": 500_000_000},
        "samples": [{"work": 1000, "duration_us": 5, "run_id": "a",
                     "repeat_index": 0, "split": "train", "source": "a"}],
    }
    fitted = fit_resource_service(packet)
    service = fitted["service"]
    assert service["rate_representation"]["input_kind"] == "nominal"
    assert service["normalization"]["input_kind"] == "nominal"
    assert service["effective_rate"] == pytest.approx(500_000_000)
    assert service["nominal_multiplier"] == pytest.approx(1)
    assert service["identification"]["status"] == "nominal_specification"
    assert service["provenance"]["parameter_class"] == "nominal_specification"
    assert service["provenance"]["normalized_rate_interpretation"] == (
        "declared_nominal_rate_not_a_calibrated_service")
    assert fitted["fit"]["declared_supply_kind"] == "nominal"
    assert normalize_resource_service(service).duration_us(2000) == pytest.approx(7)


def test_declared_startup_fits_rate_from_one_packet_without_double_counting():
    packet = {
        "id": "declared-fixed", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "primitive", "work_unit": "bytes",
        "boundary": "device_kernel", "active_units": 1,
        "startup_us": 1, "response_us": 1,
        "samples": [{"work": 1000, "duration_us": 5, "run_id": "a",
                     "repeat_index": 0, "split": "train", "source": "a"}],
    }
    fitted = fit_resource_service(packet)
    assert fitted["fit"]["rate_source"] == "fitted_with_declared_fixed"
    assert fitted["fit"]["fixed_source"] == "declared"
    assert fitted["fit"]["fixed_us"] == pytest.approx(2)
    assert fitted["fit"]["effective_rate"] == pytest.approx(1000 / 3 * 1e6)
    assert fitted["service"]["identification"]["status"] == "identified"
    assert fitted["service"]["identification"]["fixed_identified"] is False
    model = normalize_resource_service(fitted["service"])
    assert model.duration_us(2000) == pytest.approx(8)
    contradicted = deepcopy(packet)
    contradicted["samples"][0]["duration_us"] = 1.5
    with pytest.raises(ValueError, match="contradict_declared_startup"):
        fit_resource_service(contradicted)


def test_joint_copy_overlapping_busy_is_not_three_independent_rates():
    source = json.loads(JOINT_COPY.read_text())
    assert "not independent" in source["provenance"]["interpretation"]
    overlapping = service(
        id="a5-copy-joint",
        scope="independent_resource",
        resources=["copy_joint_scalar_busy", "copy_joint_mte2_busy",
                   "copy_joint_mte3_busy"],
        phase="prefill",
        work_unit="busy_ns",
        boundary="device_kernel",
        supply={"kind": "absolute_rate", "rate": 1},
    )
    with pytest.raises(ValueError, match="requires_exactly_one_resource"):
        normalize_resource_service(overlapping)
    joint = normalize_resource_service({**overlapping, "scope": "joint_phase",
                                       "identification": {
                                           "resource_identified": False}})
    assert joint.kind == "joint_phase"
    assert joint.identification["resource_identified"] is False
    assert "resource_rates" not in joint.to_dict()
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        joint.counterfactual(10, nominal_rate=1)
    fitted = fit_resource_service({
        "id": "joint-fit", "scope": "joint_phase",
        "resources": ["copy_joint_scalar_busy", "copy_joint_mte2_busy",
                      "copy_joint_mte3_busy"],
        "phase": "prefill", "work_unit": "busy_ns", "boundary": "device_kernel",
        "active_units": 64, "provenance": {"window": "joint"},
        "samples": [
            {"work": 10, "duration_us": 13, "run_id": "a", "repeat_index": 0,
             "split": "train", "source": "x"},
            {"work": 20, "duration_us": 23, "run_id": "b", "repeat_index": 0,
             "split": "train", "source": "x"},
        ],
    })
    assert fitted["service"]["kind"] == "joint_phase"
    assert fitted["service"]["identification"]["resource_identified"] is False
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        normalize_resource_service(fitted["service"]).counterfactual(
            10, nominal_rate=1)


def test_fit_result_and_formula_are_exportable_json():
    packet = a5_gather_packet()
    fitted = fit_resource_service(packet, nominal_rates=packet["nominal_rates"])
    text = json.dumps(fitted, sort_keys=True, allow_nan=False)
    restored = json.loads(text)
    assert restored["formula"].startswith("T_us =")
    assert restored["service"]["normalization"]["applied_once"] is True
    assert restored["service"]["schema"] == RESOURCE_SERVICE_SCHEMA
    assert restored["schema"] == "mindie-sim.resource-service-fit/1"
    assert "target_us" not in restored


def test_role_records_fit_only_in_sample_and_keep_transfer_separate():
    contract = {
        "scope": "independent_resource", "resources": ["hbm"],
        "phase": "copy", "work_unit": "bytes", "boundary": "resource_phase",
        "rate_scope": "aggregate", "active_units": 1,
        "provenance": {"window": "resource_phase"},
    }
    records = [
        {"cohort_id": "in-1536", "role": "in_sample", "resource": "hbm",
         "work": 1536, "resource_duration_us": 18, **contract},
        {"cohort_id": "in-3072", "role": "in_sample", "resource": "hbm",
         "work": 3072, "resource_duration_us": 34, **contract},
        {"cohort_id": "frozen-1", "role":
         "frozen_training_function_transfer_to_new_capture",
         "resource": "hbm", "work": 1536,
         "resource_duration_us": 1, **contract},
        {"cohort_id": "undeclared-1", "role": "not_declared",
         "resource": "hbm", "work": 1536,
         "resource_duration_us": 1, **contract},
    ]
    partition = calibration_role_partitions(records)
    assert partition["counts"] == {
        "in_sample": 2, "frozen_transfer": 1, "not_declared": 1}
    result = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes",
        active_units=1)
    assert result["status"] == "fitted"
    service = normalize_resource_service(result["service"])
    assert service.duration_us(1536) == pytest.approx(18)
    assert service.duration_us(3072) == pytest.approx(34)
    assert result["training_ids"] == ["in-1536", "in-3072"]
    assert len(result["training_participation"]) == 2
    assert {item["status"] for item in result["training_participation"]} == {
        "trained"}
    assert result["frozen_transfer_ids"] == ["frozen-1"]
    assert result["not_declared_ids"] == ["undeclared-1"]
    assert service.duration_us(6144) == pytest.approx(66)
    assert service.provenance["training_role"] == "in_sample"
    assert service.provenance["frozen_transfer_role"] == (
        "independent_transfer_not_in_fit")


def test_role_partition_rejects_unknown_and_duplicate_roles():
    with pytest.raises(ValueError, match="calibration_role_unknown"):
        calibration_role_partitions([
            {"cohort_id": "x", "role": "sealed_holdout"}])
    with pytest.raises(ValueError, match="duplicate_cohort_id"):
        calibration_role_partitions([
            {"cohort_id": "x", "role": "in_sample"},
            {"cohort_id": "x", "role": "not_declared"}])


def _role_contract(**changes):
    record = {
        "role": "in_sample", "resource": "hbm", "phase": "copy",
        "scope": "independent_resource", "resources": ["hbm"],
        "work_unit": "bytes", "boundary": "resource_phase",
        "rate_scope": "aggregate", "active_units": 1,
        "provenance": {"window": "resource_phase"},
    }
    record.update(changes)
    return record


def test_role_mixed_window_is_joint_evidence_not_independent_hbm():
    records = [
        {"cohort_id": f"mixed-{work}",
         **_role_contract(phase="copy", work=work, duration_us=total,
                          scope="joint_phase", boundary="device_kernel",
                          provenance={"window": "mixed"})}
        for work, total in [(1536, 18), (3072, 34)]
    ]
    independent = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes")
    assert independent["status"] == "underdetermined"
    assert independent["service"] is None
    assert len(independent["joint_evidence"]) == 2
    assert {item["window"] for item in independent["joint_evidence"]} == {"mixed"}
    assert any(item["reason"] == "joint_window_not_independent_resource"
               for item in independent["underdetermined"])
    joint = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes",
        scope="joint_phase")
    assert joint["status"] == "fitted"
    service = normalize_resource_service(joint["service"])
    assert service.kind == "joint_phase"
    assert service.identification["resource_identified"] is False
    assert service.duration_us(1536) == pytest.approx(18)
    assert service.duration_us(3072) == pytest.approx(34)
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        service.counterfactual(1536, nominal_rate=1)


def test_role_work_unit_conflict_raises_work_unit_without_conversion():
    records = [
        {"cohort_id": f"rows-{work}",
         **_role_contract(work=work, work_unit="rows",
                          resource_duration_us=total)}
        for work, total in [(1536, 18), (3072, 34)]
    ]
    with pytest.raises(ValueError, match="work_unit"):
        fit_resource_service_from_role_records(
            records, resource="hbm", phase="copy", work_unit="bytes")


def test_role_owner_scope_conflict_is_rejected_and_per_owner_kept():
    records = [
        {"cohort_id": f"owner-{work}",
         **_role_contract(work=work, rate_scope="per_owner",
                          active_units=32, resource_duration_us=total)}
        for work, total in [(1536, 18), (3072, 34)]
    ]
    with pytest.raises(ValueError, match="owner_scope_conflict"):
        fit_resource_service_from_role_records(
            records, resource="hbm", phase="copy", work_unit="bytes",
            rate_scope="aggregate")
    with pytest.raises(ValueError, match="owner_scope_conflict"):
        fit_resource_service_from_role_records(
            records, resource="hbm", phase="copy", work_unit="bytes",
            rate_scope="per_owner", active_units=1)
    fitted = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes",
        rate_scope="per_owner", active_units=32)
    service = normalize_resource_service(fitted["service"])
    assert service.rate_scope == "per_owner"
    assert service.active_units == 32
    assert service.per_owner_rate == pytest.approx(96_000_000)
    assert service.aggregate_rate == pytest.approx(96_000_000 * 32)


def test_role_duration_alias_requires_a_declared_resource_phase_window():
    records = [
        {"cohort_id": f"alias-{work}",
         **_role_contract(work=work, duration_us=total)}
        for work, total in [(1536, 18), (3072, 34)]
    ]
    missing_window = [
        {**record, "provenance": {}} for record in records
    ]
    blocked = fit_resource_service_from_role_records(
        missing_window, resource="hbm", phase="copy", work_unit="bytes")
    assert blocked["status"] == "underdetermined"
    assert blocked["service"] is None
    assert all(item["reason"] == "missing_explicit_resource_phase_work"
               for item in blocked["underdetermined"])
    mixed = [
        {**record, "provenance": {"window": "mixed"}} for record in records
    ]
    blocked = fit_resource_service_from_role_records(
        mixed, resource="hbm", phase="copy", work_unit="bytes")
    assert blocked["status"] == "underdetermined"
    assert blocked["service"] is None
    assert all(item["reason"] == "mixed_window_declared_as_independent_resource"
               for item in blocked["underdetermined"])


def test_role_source_conditions_survive_mapping_file_case_selector_and_binder(
        tmp_path):
    conditions = {"hardware": "Ascend950DT_9582", "dtype": "bf16"}
    records = [
        {"cohort_id": f"cond-{work}",
         **_role_contract(work=work, duration_us=total,
                          conditions=conditions)}
        for work, total in [(1536, 18), (3072, 34)]
    ]
    fitted = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes")
    assert fitted["status"] == "fitted"
    service = normalize_resource_service(fitted["service"])
    assert service.conditions["hardware"] == "Ascend950DT_9582"
    assert service.conditions["dtype"] == "bf16"
    assert service_conditions_match(
        service, {"hardware": "Ascend950DT_9582", "dtype": "bf16"})["applies"]
    assert not service_conditions_match(
        service, {"hardware": "Ascend950DT_9581", "dtype": "bf16"})["applies"]
    profile = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "Ascend950DT_9582", "services": [service.to_dict()],
    }
    mapping_cost = CostModel(parse_case(apply_resource_profile(case(), profile)))
    path = tmp_path / "role-conditions.json"
    export_resource_profile(profile, path=path)
    file_cost = CostModel(parse_case(apply_resource_profile(case(), path)))
    direct = case()
    direct["runtime"]["metadata"] = {"resource_service_profile": profile}
    case_cost = CostModel(parse_case(direct))
    context = {"hardware": "Ascend950DT_9582", "dtype": "bf16"}
    for cost in (mapping_cost, file_cost, case_cost):
        selected = cost.select_resource_service(context)
        by_id = cost.select_resource_service(
            context, service_id=selected.service_id)
        assert selected.model_id == by_id.model_id == service.model_id
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(
                {"hardware": "Ascend950DT_9581", "dtype": "bf16"})
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(
                {"hardware": "Ascend950DT_9581", "dtype": "bf16"},
                service_id=service.service_id)
        binding = resource_service_binding(
            selected, resource_role="hbm", phase="copy",
            work_unit="bytes", conditions=context)
        assert binding["conditions"]["hardware"] == "Ascend950DT_9582"
        with pytest.raises(ValueError, match="conditions_do_not_match"):
            resource_service_binding(
                selected, resource_role="hbm", phase="copy",
                work_unit="bytes",
                conditions={"hardware": "Ascend950DT_9581", "dtype": "bf16"})


def test_role_frozen_and_undeclared_values_never_move_training_parameters():
    records = [
        {"cohort_id": "in-1536",
         **_role_contract(work=1536, resource_duration_us=18)},
        {"cohort_id": "in-3072",
         **_role_contract(work=3072, resource_duration_us=34)},
        {"cohort_id": "frozen-1",
         **_role_contract(work=1536, resource_duration_us=1),
         "role": "frozen_training_function_transfer_to_new_capture"},
        {"cohort_id": "undeclared-1",
         **_role_contract(work=3072, resource_duration_us=1),
         "role": "not_declared"},
    ]
    first = fit_resource_service_from_role_records(
        records, resource="hbm", phase="copy", work_unit="bytes")
    changed = deepcopy(records)
    changed[2]["resource_duration_us"] = 10_000
    changed[3]["resource_duration_us"] = 20_000
    second = fit_resource_service_from_role_records(
        changed, resource="hbm", phase="copy", work_unit="bytes")
    a = normalize_resource_service(first["service"])
    b = normalize_resource_service(second["service"])
    assert a.model_id == b.model_id
    assert a.fixed_us == pytest.approx(2)
    assert a.effective_rate == pytest.approx(96_000_000)
    assert a.duration_us(6144) == pytest.approx(66)
