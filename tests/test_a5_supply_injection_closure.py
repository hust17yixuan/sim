"""C03 supply injection: normalized services production, legacy replay diagnostic."""
from copy import deepcopy
from pathlib import Path

import pytest

from mindie_sim import predict
from mindie_sim.compose import compose_case
from mindie_sim.observations.calibration import (
    apply_profiler_supply, apply_resource_profile, bind_operator_resource_service,
    export_resource_profile, import_profiler_supply, load_resource_profile,
    resource_service_binding, select_resource_service_candidate,
)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_BINDING_SCHEMA, RESOURCE_SERVICE_PROFILE_SCHEMA,
    RESOURCE_SERVICE_SCHEMA, normalize_resource_service,
    select_resource_service, sparse_mte1_bytes,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from test_prediction_system import case


HERE = Path(__file__).parent
SUPPLY = HERE / "fixtures" / "r64_qbmm_dual_load" / "supply.json"
HARDWARE = "Ascend950DT_9582"


def _composed():
    return compose_case(
        "glm-5.1", HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
        prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
        num_devices=1, ranks_per_node=1,
    )


def test_ordinary_supply_attaches_normalized_services_not_legacy_kernels():
    imported = import_profiler_supply(
        SUPPLY, recipe="output", hardware=HARDWARE, owners=24)
    assert imported["service_regimes"][0]["execution"]["kind"] == "mac_service"
    normal = apply_profiler_supply(
        _composed(), SUPPLY, recipe="output", hardware=HARDWARE, owners=24)
    kernels = normal["runtime"].get("kernels", {})
    assert "service_regimes" not in kernels
    metadata = normal["runtime"]["metadata"]
    measured = metadata["profiler_measured_supply"]
    assert measured["legacy_service_regimes"]
    assert all(rule["production_prediction"] is False
               for rule in measured["legacy_service_regimes"])
    assert measured["legacy_diagnostics_replay"] is False
    assert measured["source"]["kind"] == "file"
    assert measured["source"]["path"] == str(SUPPLY.resolve())
    assert measured["diagnostic_comparison"]["production_prediction"] is False
    profile = metadata["resource_service_profile"]
    assert profile["schema"] == RESOURCE_SERVICE_PROFILE_SCHEMA
    assert profile["services"]
    normalized = [normalize_resource_service(item) for item in profile["services"]]
    assert measured["diagnostic_comparison"]["normalized_parameters"][0][
        "service_id"] == normalized[0].service_id
    assert normalized[0].resource == "mte1"
    assert normalized[0].rate_scope == "per_owner"
    assert normalized[0].active_units == 32
    assert normalized[0].effective_rate == pytest.approx(346958503130.8095)
    assert normalized[0].aggregate_rate == pytest.approx(
        346958503130.8095 * 32)
    assert normalized[0].provenance["source_rate_scope"] == "critical_owner"
    assert normalized[0].provenance["rate_conversion"] == (
        "per_owner_rate_preserved_critical_owner_work")
    assert normalized[0].conditions["hardware"] == HARDWARE
    assert normalized[0].conditions["dtype"] == "bf16"
    assert normalized[0].conditions["native_type"] == (
        "KvQuantSparseFlashAttention")
    assert normalized[0].conditions["phase"] == "prefill"
    assert normalized[0].conditions["op_state"] == "dynamic"
    assert normalized[0].conditions["owners"] == 32
    assert normalized[0].conditions["source_identity.source_scope"] == "r64"
    assert normalized[0].conditions["geometry.q_len"] == {"min": 1024}
    assert normalized[0].optional_conditions["mix_block_num"] == 64
    assert normalized[0].provenance["selection_policy"] == "c7_applicability_match"
    cost = CostModel(parse_case(normal))
    context = {
        "hardware": HARDWARE, "dtype": "bf16",
        "native_type": "KvQuantSparseFlashAttention",
        "phase": "prefill", "op_state": "dynamic", "owners": 32,
        "source_identity": {
            "source_scope": "r64",
            "image_digest": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
        },
        "geometry": {"q_len": 1024}, "mix_block_num": 64,
    }
    selected = cost.select_resource_service(context)
    assert selected.service_id == normalized[0].service_id
    assert cost.select_resource_service(
        context, service_id=selected.service_id).service_id == selected.service_id

    replay = apply_profiler_supply(
        _composed(), SUPPLY, recipe="output", hardware=HARDWARE, owners=24,
        legacy_diagnostics=True)
    replay_rules = replay["runtime"]["kernels"]["service_regimes"]
    assert replay_rules
    assert "production_prediction" not in replay_rules[0]
    assert replay["runtime"]["metadata"]["profiler_measured_supply"][
        "legacy_service_regimes"][0]["production_prediction"] is False
    assert replay["runtime"]["metadata"]["profiler_measured_supply"][
        "legacy_diagnostics_replay"] is True
    assert "service_regimes" not in _composed()["runtime"]
    ordinary = predict(normal, scope="body")
    assert "service_regimes" not in ordinary["parameters"]
    assert ordinary["parameters"]["production_price_basis"] == "physical_work_only"


def test_legacy_replay_and_ordinary_case_keep_the_source_rule_recoverable():
    normal = apply_profiler_supply(
        _composed(), SUPPLY, recipe="output", hardware=HARDWARE, owners=24)
    measured = normal["runtime"]["metadata"]["profiler_measured_supply"]
    rule = measured["legacy_service_regimes"][0]
    assert rule["id"]
    assert rule["id"].startswith("measured-dual-load-")
    assert rule["diagnostic_source"]["kind"] == "file"
    assert rule["source_identity"]["source_scope"] == "r64"
    assert rule["diagnostic_only"] is True
    assert measured["diagnostic_comparison"]["rules"][0]["rule_id"] == rule["id"]
    # The raw report resource row remains readable and is not rewritten.
    assert measured["resource_services"][0]["rate_bytes_per_s"] == pytest.approx(
        346958503130.8095)
    assert measured["normalized_resource_services"][0]["schema"] == RESOURCE_SERVICE_SCHEMA


def test_generated_binding_is_applied_once_by_the_real_t04_program_path():
    record = {
        "id": "svc", "scope": "independent_resource", "resources": ["hbm"],
        "phase": "decode", "work_unit": "bytes", "boundary": "resource_phase",
        "startup_us": 1000, "response_us": 0, "rate_scope": "aggregate",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 1_000_000},
    }
    service = normalize_resource_service(record)
    binding = resource_service_binding(
        service, resource_role="hbm", phase="decode", work_unit="bytes")
    assert binding["schema"] == RESOURCE_SERVICE_BINDING_SCHEMA
    assert binding["service_id"] == "svc"
    op = OpSpec("copy", "copy", "decode", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16", recipe="copy")
    bound = bind_operator_resource_service(
        op, service, resource_role="hbm", phase="decode", work_unit="bytes")
    assert dict(bound.metadata)["resource_service"]["service_id"] == "svc"
    assert isinstance(bound, OpSpec)
    again = bind_operator_resource_service(
        bound, service, resource_role="hbm", phase="decode", work_unit="bytes")
    assert (dict(again.metadata)["resource_service"]
            == dict(bound.metadata)["resource_service"])
    with pytest.raises(ValueError, match="work_unit_mismatch"):
        resource_service_binding(service, resource_role="hbm",
                                 phase="decode", work_unit="bytes_total")
    with pytest.raises(ValueError, match="phase_mismatch"):
        resource_service_binding(service, resource_role="hbm",
                                 phase="prefill", work_unit="bytes")

    from types import SimpleNamespace
    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    execution.resource_services = {"svc": service}
    context = SimpleNamespace(
        op=SimpleNamespace(metadata=(("resource_service", binding),)),
        resource_names={"hbm": "memory"}, work_units={"hbm": "bytes"})
    wrapped = execution._apply_resource_service_program(
        context, KernelProgram((Stage("work", {"memory": 1000}),)))
    engine = Engine([Resource("memory", 1000, unit="bytes")])
    ready = engine.submit("ready")
    done = wrapped.start(engine, "svc", ready)
    engine.advance(until=(done,))
    assert done.time == 2
    assert engine.serviced == {"memory": 1000.0}


def test_requested_hardware_mismatch_is_a_selection_gap_not_profile_conflict():
    bound = apply_profiler_supply(
        _composed(), SUPPLY, recipe="output", hardware="Ascend910_9392",
        owners=24)
    measured = bound["runtime"]["metadata"]["profiler_measured_supply"]
    assert measured["applicability_gaps"][0]["gap"] == "hardware_mismatch"
    profile = bound["runtime"]["metadata"]["resource_service_profile"]
    # The profile retains the source hardware; the caller's different placement
    # is checked by the shared selector, not broadcast into service conditions.
    assert profile["hardware"] == HARDWARE
    cost = CostModel(parse_case(bound))
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        cost.select_resource_service({
            "hardware": "Ascend910_9392", "dtype": "bf16",
            "geometry": {"shape_class":
                         "prefill_causal_q1024_g64_dq576_dv512_s2tile128",
                         "q_len": 1024, "owners": 32},
        })


def _raw_c7():
    import json
    return json.loads(SUPPLY.read_text())["resource_services"][0]


def _c7_context(**overrides):
    context = {
        "hardware": HARDWARE,
        "dtype": "bf16",
        "native_type": "KvQuantSparseFlashAttention",
        "phase": "prefill",
        "op_state": "dynamic",
        "owners": 32,
        "source_identity": {
            "source_scope": "r64",
            "image_digest": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
        },
        "geometry": {"q_len": 1024},
        "mix_block_num": 64,
    }
    context.update(overrides)
    return context


def _c7_normal_case():
    return apply_profiler_supply(
        _composed(), SUPPLY, recipe="output", hardware=HARDWARE, owners=24)


def _raw_select(context, q_len=1024, **overrides):
    kwargs = {
        "native_type": context["native_type"],
        "hardware": context["hardware"],
        "source_identity": context.get("source_identity"),
        "dtype": context["dtype"],
        "q_len": q_len,
        "context": 0,
        "kv_len": q_len,
        "causal": True,
        "owners": context["owners"],
        "phase": context["phase"],
        "op_state": context["op_state"],
        "mix_block_num": context.get("mix_block_num"),
    }
    kwargs.update(overrides)
    return select_resource_service([_raw_c7()], **kwargs)


def test_c7_applicability_is_shared_by_raw_normalized_and_direct_binding():
    normal = _c7_normal_case()
    cost = CostModel(parse_case(normal))
    context = _c7_context()
    raw = _raw_select(context)
    assert raw is not None
    auto = cost.select_resource_service(context)
    by_id = cost.select_resource_service(context, service_id=auto.service_id)
    assert auto.service_id == by_id.service_id
    direct = resource_service_binding(
        auto, resource_role="mte1", phase=auto.phase,
        work_unit=auto.work_unit, conditions=context)
    assert direct["service_id"] == auto.service_id

    wrong = {
        "missing_source": {"source_identity": None},
        "wrong_source": {"source_identity": {
            "source_scope": "r64", "image_digest": "f" * 64}},
        "wrong_native": {"native_type": "MatMul"},
        "wrong_state": {"op_state": "static"},
        "wrong_owner": {"owners": 24},
        "wrong_mix": {"mix_block_num": 32},
    }
    for label, override in wrong.items():
        ctx = _c7_context(**override)
        assert _raw_select(ctx) is None, label
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(ctx)
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(ctx, service_id=auto.service_id)
        with pytest.raises(ValueError, match="conditions_do_not_match|conditions_have_no"):
            resource_service_binding(
                auto, resource_role="mte1", phase=auto.phase,
                work_unit=auto.work_unit, conditions=ctx)

    unrelated = _c7_context(unrelated_new_field={"ignored": True})
    unrelated.pop("mix_block_num")
    assert _raw_select(unrelated) is not None
    assert cost.select_resource_service(unrelated).service_id == auto.service_id


def test_c7_saturation_q2048_uses_geometry_work_not_exact_q_condition():
    normal = _c7_normal_case()
    cost = CostModel(parse_case(normal))
    ctx1024 = _c7_context()
    ctx2048 = _c7_context(geometry={"q_len": 2048})
    assert _raw_select(ctx1024, q_len=1024) is not None
    assert _raw_select(ctx2048, q_len=2048) is not None
    s1024 = cost.select_resource_service(ctx1024)
    s2048 = cost.select_resource_service(ctx2048)
    assert s1024.service_id == s2048.service_id
    g1024 = sparse_mte1_bytes(1024, 0, 1024, owners=32)
    g2048 = sparse_mte1_bytes(2048, 0, 2048, owners=32)
    assert g2048["critical_qk_pv_bytes"] > g1024["critical_qk_pv_bytes"] > 0
    assert s2048.rate_scope == "per_owner"
    assert s2048.duration_us(
        g2048["critical_qk_pv_bytes"], work_scope="per_owner") == pytest.approx(
        g2048["critical_qk_pv_bytes"] * 1e6 / s2048.effective_rate)


def test_c7_conditions_survive_export_file_and_case_reload(tmp_path):
    normal = _c7_normal_case()
    profile = normal["runtime"]["metadata"]["resource_service_profile"]
    context = _c7_context()
    direct = CostModel(parse_case(normal)).select_resource_service(context)
    loaded = load_resource_profile(profile)
    loaded_model = normalize_resource_service(loaded["services"][0])
    assert loaded_model.conditions["native_type"] == "KvQuantSparseFlashAttention"
    assert loaded_model.conditions["geometry.q_len"] == {"min": 1024}
    assert loaded_model.optional_conditions["mix_block_num"] == 64
    path = tmp_path / "c7-profile.json"
    export_resource_profile(profile, path=path)
    from_file = load_resource_profile(path)
    from_file_model = normalize_resource_service(from_file["services"][0])
    assert from_file_model.conditions == loaded_model.conditions
    assert from_file_model.optional_conditions == loaded_model.optional_conditions
    by_file = CostModel(parse_case(apply_resource_profile(_composed(), path)))
    assert by_file.select_resource_service(context).service_id == direct.service_id
    by_mapping = CostModel(parse_case(apply_resource_profile(_composed(), profile)))
    assert by_mapping.select_resource_service(context).service_id == direct.service_id


def test_c7_matched_binding_enters_t04_and_wrong_source_does_not():
    normal = _c7_normal_case()
    profile_service = normalize_resource_service(
        normal["runtime"]["metadata"]["resource_service_profile"]["services"][0])
    service = normalize_resource_service({
        **profile_service.to_dict(),
        "id": "c7-matched-t04",
        "resources": ["hbm"],
        "effective_rate": 1_000_000,
        "aggregate_rate": 1_000_000,
        "per_owner_rate": 1_000_000,
        "active_units": 1,
        "rate_scope": "aggregate",
        "startup_us": 1000,
        "response_us": 0,
        "fixed_us": 1000,
        "supply": {"kind": "absolute_rate", "rate": 1_000_000},
    })
    profile = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": HARDWARE,
        "services": [service.to_dict()],
    }
    raw = apply_resource_profile(_composed(), profile)
    cost = CostModel(parse_case(raw))
    context = _c7_context()
    op = OpSpec("copy", "copy", "prefill", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16", recipe="copy")
    bound = cost.bind_operator_resource_service(
        op, context, resource_role="hbm", phase="prefill")
    binding = dict(bound.metadata)["resource_service"]
    assert binding["service_id"] == "c7-matched-t04"
    execution = Execution(parse_case(raw), cost, include_submission=False)
    execution.resource_services = {"c7-matched-t04": service}
    from types import SimpleNamespace
    ctx = SimpleNamespace(
        op=SimpleNamespace(metadata=tuple(bound.metadata)),
        resource_names={"hbm": "memory"}, work_units={"hbm": "bytes"})
    wrapped = execution._apply_resource_service_program(
        ctx, KernelProgram((Stage("work", {"memory": 1000}),)))
    engine = Engine([Resource("memory", 1000, unit="bytes")])
    ready = engine.submit("ready")
    done = wrapped.start(engine, "c7", ready)
    engine.advance(until=(done,))
    assert done.time == pytest.approx(2)
    assert engine.serviced == {"memory": 1000.0}
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        cost.bind_operator_resource_service(
            op, _c7_context(source_identity=None), resource_role="hbm",
            phase="prefill")


def test_real_c7_service_is_consumed_once_by_an_actual_stage_with_receipt():
    """The accepted C7 per-owner MTE1 price feeds one real Stage unchanged."""
    normal = _c7_normal_case()
    cost = CostModel(parse_case(normal))
    context = _c7_context()
    service = cost.select_resource_service(context)
    assert service.resource == "mte1"
    assert service.rate_scope == "per_owner"
    assert service.active_units == 32
    critical = sparse_mte1_bytes(
        1024, 0, 1024, owners=32)["critical_qk_pv_bytes"]
    op = OpSpec("mte1-service", "mte1_service", "prefill", 0, 0, ("r",),
                (1,), (1,), (1,), (1,), mte2_bytes=critical, dtype="bf16",
                recipe="mte1_service")
    bound = cost.bind_operator_resource_service(
        op, context, resource_role="mte1", phase="prefill")
    binding = dict(bound.metadata)["resource_service"]
    assert binding["service_id"] == service.service_id
    assert binding["work_unit"] == "bytes"
    assert binding["rate_scope"] == "per_owner"
    assert binding["normalization"]["applied_once"] is True
    execution = Execution(parse_case(normal), cost, include_submission=False)
    execution.resource_services = {service.service_id: service}
    from types import SimpleNamespace
    stage_context = SimpleNamespace(
        op=SimpleNamespace(metadata=tuple(bound.metadata)),
        resource_names={"mte1": "mte1"}, work_units={"mte1": "bytes"})
    program = execution._apply_resource_service_program(
        stage_context, KernelProgram((Stage("mte1_compute", {"mte1": critical}),)))
    engine = Engine([Resource("mte1", service.per_owner_rate / 1000.0,
                              unit="bytes", policy="fifo")])
    ready = engine.submit("ready")
    done = program.start(engine, "c7-real-service", ready)
    engine.advance(until=(done,))
    expected_us = service.duration_us(critical, work_scope="per_owner")
    receipt = {
        "schema": "mindie-sim/resource-service-receipt/1",
        "service_id": service.service_id,
        "model_id": service.model_id,
        "source": service.provenance.get("source"),
        "source_sha256": service.provenance.get("source_sha256"),
        "binding": binding,
        "work": critical,
        "work_unit": "bytes",
        "rate_scope": "per_owner",
        "engine_serviced": dict(engine.serviced),
        "duration_us": done.time * 1000.0,
        "expected_us": expected_us,
        "fixed_once": True,
        "startup_once": True,
        "conditions_checked": True,
    }
    assert receipt["engine_serviced"] == {"mte1": float(critical)}
    assert receipt["duration_us"] == pytest.approx(
        receipt["expected_us"], rel=1e-12, abs=1e-9)
    assert receipt["source_sha256"] == service.provenance["source_sha256"]
    assert receipt["binding"]["conditions"] == service.conditions
    assert receipt["binding"]["fixed_once"] is True
