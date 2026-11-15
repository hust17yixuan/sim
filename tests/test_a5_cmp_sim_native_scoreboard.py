"""Independent native comparison slices retain per-item evidence and denominators."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from mindie_sim.cohort_accuracy import native_acceptance, native_acceptance_policy
from mindie_sim.coverage import record_linkage
from mindie_sim.observations.accuracy import accuracy_report
from mindie_sim.observations.scoreboard import main, native_comparison_scoreboard
from mindie_sim.observations.scoreboard_accuracy import render_native_comparison_markdown


JOBS = Path(os.environ.get(
    "A5_JOBS_ROOT", "/Volumes/data/Codex/2026-11-12/kan-x/work/a5-implementation/jobs"))
T16_SLICE = JOBS / "A5-T16/example_comparison.json"
T16_WORK_ONLY = JOBS / "A5-T16/example_t12_comparison.json"
T16_CONTRACT = JOBS / "A5-CMP-SIM/out/t16_contract"
T03 = JOBS / "A5-T03/out"
T03_INVENTORY = T03 / "inventory"


def _sample(observed, predicted, identity, diagnostic=None):
    residual = predicted - observed
    relative = residual / observed
    formal_allowed = 0.1 * observed
    sample = {
        "absolute_residual_ms": abs(residual),
        "formal_allowed_error_ms": formal_allowed,
        "clock_domain": "device:0:stream:7",
        "observation_id": identity,
        "observed_ms": observed,
        "relative_error": relative,
        "signed_residual_ms": residual,
        "within_tolerance": abs(residual) <= formal_allowed,
    }
    if diagnostic is not None:
        diagnostic_allowed = (diagnostic["absolute_ms"]
                              + diagnostic["relative"] * observed)
        sample.update({
            "diagnostic_allowed_error_ms": diagnostic_allowed,
            "diagnostic_within": abs(residual) <= diagnostic_allowed,
        })
    return sample


def _item(observed, predicted, suffix, gaps=None, diagnostic=None):
    abi = "sha256:" + suffix * 64
    sample = _sample(observed, predicted, "observation:" + suffix, diagnostic)
    residual = predicted - observed
    relative = residual / observed
    within = abs(residual) <= 0.1 * observed
    return {
        "item_id": "comparison:" + suffix,
        "request_id": "request:" + suffix,
        "prediction_id": "prediction:" + suffix,
        "identity": {
            "native_operator": "QuantBatchMatmulV3",
            "native_abi_sha256": abi,
            "hardware": "Ascend950DT_9582", "phase": "prefill",
            "model": "glm-5.1", "quant": "W8A8_MXFP8",
        },
        "boundary": {
            "kind": "kernel", "scope": "one_complete_kernel_event",
            "fusion": {"members": [], "status": "not_evidenced"},
        },
        "observation_abi_digest": abi,
        "prediction_abi_digest": abi,
        "observed_ms": observed,
        "predicted_ms": predicted,
        "observation_samples": 1,
        "observation_ids": ["observation:" + suffix],
        "observation_range_ms": [observed, observed],
        "condition_gaps": list(gaps or ()),
        "comparison": {
            "signed_residual_ms": residual,
            "absolute_residual_ms": abs(residual),
            "relative_error": relative,
            "formal_allowed_error_ms": 0.1 * observed,
            "formal_policy": "fixed_A5_formal_itemwise_10pct_no_absolute_allowance",
            "denominator": "observed_ms",
            "within_tolerance": within,
            "sample_verdicts": [sample],
            "outside_sample_ids": [] if within else [sample["observation_id"]],
            "worst_absolute_relative_error": abs(relative),
        },
        "status": "within_tolerance" if within else "outside_tolerance",
    }


def _work_only_item(observed, suffix, macs):
    """Synthetic T16 work-only item: ABI pair plus work/observation facts."""
    abi = "sha256:" + suffix * 64
    return {
        "item_id": "comparison:" + suffix,
        "request_id": "request:" + suffix,
        "prediction_id": "prediction:" + suffix,
        "identity": {
            "native_operator": "QuantBatchMatmulV3",
            "native_abi_sha256": abi,
            "hardware": "Ascend950DT_9582", "phase": "prefill",
            "model": "glm-5.1", "quant": "W8A8_MXFP8",
        },
        "boundary": {
            "kind": "kernel", "scope": "one_complete_kernel_event",
            "fusion": {"members": [], "status": "not_evidenced"},
        },
        "observation_abi_digest": abi,
        "prediction_abi_digest": abi,
        "observed_ms": observed,
        "predicted_ms": None,
        "observation_samples": 1,
        "observation_ids": ["observation:" + suffix],
        "observation_range_ms": [observed, observed],
        "observation_window_ids": ["window:" + suffix],
        "condition_gaps": [],
        "observation_evidence": {"pipe_times_us": [], "block_num": [24]},
        "prediction_evidence": ["test.work_only.no_duration"],
        "comparison": {
            "duration_available": False,
            "formal_policy": "not_evaluated_no_predicted_duration",
            "reason": "prediction_work_only",
            "work": {
                "cube": {"macs": macs, "flops": 2 * macs,
                         "issued_macs": macs, "issued_basis": "test"},
                "vector": {"ops": 0, "scale_ops": 0, "bias_ops": 0},
                "bytes": {"total": 1024, "basis": "test"},
                "call_boundary": {
                    "tensor_footprint_bytes": 1024, "issued_traffic": None,
                    "issued_traffic_status": "requires_selected_load_store_plan",
                },
            },
            "observation_duration_ms": observed,
            "sample_verdicts": [],
        },
        "status": "work_only",
    }


def _report(items, unpaired_observations=(), unpaired_predictions=(), ambiguous=(),
            *, partial=False, truncated=False, diagnostic=None):
    items = list(items)
    unpaired_observations = list(unpaired_observations)
    unpaired_predictions = list(unpaired_predictions)
    ambiguous = list(ambiguous)
    within = sum(item["status"] == "within_tolerance" for item in items)
    outside = sum(item["status"] == "outside_tolerance" for item in items)
    work_only = sum(item["status"] == "work_only" for item in items)
    timed = [item for item in items if item["status"] != "work_only"]
    samples = sum(len(item["comparison"]["sample_verdicts"]) for item in timed)
    sample_within = sum(
        row["within_tolerance"]
        for item in timed for row in item["comparison"]["sample_verdicts"])
    failure_matrix = [{
        "root_cause": "unclassified_residual",
        "reason": "relative_error_exceeds_tolerance",
        "count": 1,
        "item_ids": [item["item_id"]],
        "claim": "candidate_dimension_not_proven_root_cause",
        "evidence": {},
    } for item in items if item["status"] == "outside_tolerance"]
    for item in items:
        if item["status"] == "work_only":
            failure_matrix.append({
                "root_cause": "prediction_gap",
                "reason": "prediction_work_only_no_duration",
                "count": 1, "item_ids": [item["item_id"]],
                "claim": "candidate_dimension_not_proven_root_cause", "evidence": {},
            })
    for row in unpaired_observations:
        failure_matrix.append({
            "root_cause": "evidence_missing", "reason": row["reason"],
            "count": 1, "item_ids": [row["observation_ids"][0]],
            "claim": "candidate_dimension_not_proven_root_cause", "evidence": {},
        })
    if partial or truncated:
        failure_matrix.append({
            "root_cause": "evidence_missing",
            "reason": "partial_observation_window_coverage",
            "count": 1, "item_ids": ["coverage"],
            "claim": "candidate_dimension_not_proven_root_cause", "evidence": {},
        })
    status = ("failed" if outside
              else "inconclusive" if unpaired_observations or unpaired_predictions
              or ambiguous or work_only or not items
              else "partial" if partial or truncated else "passed")
    observed_rows = (sum(item["observation_samples"] for item in items)
                     + sum(row.get("observation_samples", 0)
                           for row in unpaired_observations))
    window = {
        "kind": ("truncated_local_domain_range" if truncated
                 else "local_domain_range" if len(items) == 1 else "per_domain"),
        "count": len(items),
        "clock_comparable_across_domains": False,
        "windows": [],
    }
    reconciled = not (partial or truncated)
    coverage = {
        "observations": observed_rows,
        "predictions": len(items) + len(unpaired_predictions) + sum(
            len(row.get("predictions", ())) for row in ambiguous),
        "request_groups": len(items) + len(unpaired_observations),
        "truncated": bool(truncated),
        "partial_coverage": bool(partial or truncated),
        "window": window,
        "windows": [],
        "source_identity": {
            "path": "/tmp/synthetic-kernel_details.csv",
            "sha256": "sha256:" + "ab" * 32,
            "sha256_scope": "raw_file_bytes",
        },
        "observation_counts": {
            "events": observed_rows, "confirmed": observed_rows,
            "candidate": 0, "unidentified": 0,
        },
        "reconciliation": {
            "reconciled": reconciled,
            "subset": False,
            "scope": None,
            "parent_capture": None,
            "declared_window_counts": {},
            "actual_window_counts": {},
            "missing_declared_window_ids": [],
            "missing_window_ids": [],
            "unexpected_window_ids": [],
            "declared_events": observed_rows,
            "actual_events": observed_rows,
            "missing_events": 0,
            "unexpected_events": 0,
            "truncated": bool(truncated),
        },
    }
    return {
        "schema": "mindie-profiler-native-comparison/1",
        "status": status,
        "tolerance": {
            "formal": {
                "relative": 0.1, "absolute_ms": 0.0,
                "denominator": "observed_ms",
                "policy": "fixed_A5_formal_itemwise_10pct_no_absolute_allowance",
            },
            "diagnostic": diagnostic,
            "applies_to": "paired_comparable_items_only",
        },
        "boundary_policy": {
            "join": "native_operator+native_abi_sha256+boundary+declared_conditions",
            "name_only_allowed": False,
            "family_average_allowed": False,
            "fusion_member_allocation_allowed": False,
        },
        "coverage": coverage,
        "counts": {
            "observations": observed_rows,
            "predictions": coverage["predictions"],
            "request_groups": coverage["request_groups"],
            "paired": len(items),
            "within_tolerance": within,
            "outside_tolerance": outside,
            "work_only": work_only,
            "observation_verdicts": samples,
            "observations_within_tolerance": sample_within,
            "observations_outside_tolerance": samples - sample_within,
            "items_with_condition_gaps": sum(bool(item["condition_gaps"]) for item in items),
            "unpaired_observations": len(unpaired_observations),
            "unpaired_predictions": len(unpaired_predictions),
            "ambiguous": len(ambiguous),
            "truncated": bool(truncated),
            "partial_coverage": bool(partial or truncated),
            "family_aggregates_used": 0,
            "name_only_matches": 0,
            "fused_members_allocated": 0,
        },
        "items": items,
        "unpaired_observations": unpaired_observations,
        "unpaired_predictions": unpaired_predictions,
        "ambiguous": ambiguous,
        "failure_matrix": failure_matrix,
        "non_claims": ["no_family_average_or_signed_cancellation_pass"],
        "interpretation": "synthetic current-contract report",
    }


def _unpaired_observation(identity):
    return {
        "request_id": "request:" + identity,
        "observation_ids": ["observation:" + identity],
        "observed_ms": 0.123, "observation_samples": 1,
        "identity": {"native_operator": "DynamicMxQuant"},
        "boundary": {"kind": "kernel"},
        "reason": "no_exact_prediction_identity",
    }


def _unpaired_prediction(identity):
    return {
        "prediction_id": "prediction:" + identity,
        "request_id": "request:" + identity,
        "predicted_ms": 0.456,
        "identity": {"native_operator": "DynamicMxQuant"},
        "boundary": {"kind": "kernel"},
        "status": "modeled",
        "reason": "no_exact_observation_identity",
    }


def test_work_only_item_is_retained_without_a_duration_verdict():
    report = _report([_work_only_item(0.05, "w", 100663296)])
    result = native_acceptance(report)
    assert result["source_status"] == "inconclusive"
    assert result["status"] == "incomplete"
    assert result["counts"]["paired_items"] == 1
    assert result["counts"]["work_only"] == 1
    assert result["counts"]["duration_denominator_items"] == 0
    assert result["counts"]["within"] == 0
    assert result["counts"]["outside"] == 0
    item = result["items"][0]
    assert item["status"] == "work_only"
    assert item["predicted_ms"] is None
    assert item["observed_ms"] == 0.05
    assert item["samples"] == []
    assert item["work"]["cube"]["macs"] == 100663296
    board = native_comparison_scoreboard(report, label="work-only")
    accuracy = board["denominators"]["accuracy"]
    assert board["status"] == "incomplete"
    assert board["source_status"] == "inconclusive"
    assert accuracy["paired_items"] == 1
    assert accuracy["duration_denominator_items"] == 0
    assert accuracy["work_only"] == 1
    assert board["cohorts"][0]["work_only"] == 1
    assert board["cohorts"][0]["predicted_ms_range"] is None
    assert board["cohorts"][0]["worst_absolute_relative_error"] is None
    markdown = render_native_comparison_markdown(board)
    assert "Work-only items (no duration verdict)" in markdown
    assert "100663296" in markdown
    assert "prediction_work_only" in markdown


def test_work_only_cannot_carry_a_predicted_duration_or_verdict():
    report = _report([_work_only_item(0.05, "w", 100663296)])
    report["items"][0]["predicted_ms"] = 0.05
    with pytest.raises(ValueError, match="work_only_predicted_ms_must_stay_null"):
        native_acceptance(report)
    report = _report([_work_only_item(0.05, "w", 100663296)])
    report["items"][0]["comparison"]["within_tolerance"] = True
    with pytest.raises(ValueError, match="work_only_forbids_scored_field:within_tolerance"):
        native_acceptance(report)
    report = _report([_work_only_item(0.05, "w", 100663296)])
    report["items"][0]["comparison"]["sample_verdicts"] = [_sample(0.05, 0.05, "w")]
    with pytest.raises(ValueError, match="work_only_must_not_retain_sample_verdicts"):
        native_acceptance(report)
    report = _report([_work_only_item(0.05, "w", 100663296)])
    report["counts"]["work_only"] = 0
    with pytest.raises(ValueError, match="count_mismatch:work_only"):
        native_acceptance(report)


def test_mixed_within_and_work_only_uses_one_item_duration_denominator(tmp_path):
    report = _report([_item(0.1, 0.105, "a"),
                      _work_only_item(0.2, "w", 103079215104)])
    source = tmp_path / "mixed-within.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(source), "--output", str(tmp_path / "mixed-within-out.json")]) == 4
    result = native_acceptance(report)
    assert result["source_status"] == "inconclusive"
    assert result["status"] == "incomplete"
    assert result["counts"]["paired_items"] == 2
    assert result["counts"]["duration_denominator_items"] == 1
    assert result["counts"]["within"] == 1
    assert result["counts"]["outside"] == 0
    assert result["counts"]["work_only"] == 1
    board = native_comparison_scoreboard(report, label="mixed-within")
    accuracy = board["denominators"]["accuracy"]
    assert board["status"] == "incomplete"
    assert accuracy["paired_items"] == 2
    assert accuracy["duration_denominator_items"] == 1
    assert (accuracy["within"], accuracy["outside"]) == (1, 0)
    accuracy_report_value = accuracy_report(report)
    assert accuracy_report_value["status"] == "incomplete"
    assert accuracy_report_value["coverage"]["paired_items"] == 2
    assert accuracy_report_value["coverage"]["duration_denominator_items"] == 1
    assert accuracy_report_value["tolerance_outcomes"]["work_only"] == 1
    points = {point["status"] for point in accuracy_report_value["points"]}
    assert points == {"paired", "work_only"}


def test_mixed_outside_and_work_only_stays_failed_with_one_denominator(tmp_path):
    report = _report([_item(0.1, 0.13, "b"),
                      _work_only_item(0.2, "w", 103079215104)])
    source = tmp_path / "mixed-outside.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(source), "--output", str(tmp_path / "mixed-outside-out.json")]) == 3
    result = native_acceptance(report)
    assert result["source_status"] == "failed"
    assert result["status"] == "failed"
    assert result["counts"]["paired_items"] == 2
    assert result["counts"]["duration_denominator_items"] == 1
    assert result["counts"]["within"] == 0
    assert result["counts"]["outside"] == 1
    assert result["counts"]["work_only"] == 1
    board = native_comparison_scoreboard(report, label="mixed-outside")
    assert board["status"] == "failed"
    assert board["denominators"]["accuracy"]["duration_denominator_items"] == 1
    assert board["denominators"]["accuracy"]["outside"] == 1


def test_unpaired_and_ambiguous_sides_stay_out_of_accuracy_denominator():
    report = _report(
        [_item(0.1, 0.105, "a")],
        unpaired_observations=[_unpaired_observation("b")],
        unpaired_predictions=[_unpaired_prediction("c")],
        ambiguous=[{"request_id": "request:d", "reason": "ambiguous_predictions",
                    "predictions": ["prediction:d1", "prediction:d2"]}],
    )
    board = native_comparison_scoreboard(report, label="synthetic")
    accuracy = board["denominators"]["accuracy"]
    assert board["status"] == "incomplete"
    assert accuracy["paired_items"] == 1
    assert accuracy["within"] == 1 and accuracy["outside"] == 0
    assert accuracy["unpaired_observations"] == 1
    assert accuracy["unpaired_predictions"] == 1
    assert accuracy["ambiguous"] == 1
    assert board["structure_accuracy"]["computed"] is False
    assert board["unpaired_observations"][0]["reason"] == "no_exact_prediction_identity"
    assert board["unpaired_predictions"][0]["reason"] == "no_exact_observation_identity"
    markdown = render_native_comparison_markdown(board)
    assert "no_exact_prediction_identity" in markdown
    assert "no_exact_observation_identity" in markdown


def test_formal_tolerance_cannot_be_relaxed_or_renamed():
    report = _report([_item(0.1, 0.105, "a")])
    report["tolerance"]["formal"]["relative"] = 0.15
    with pytest.raises(ValueError, match="formal_relative_tolerance_must_stay"):
        native_acceptance(report)
    report = _report([_item(0.1, 0.105, "a")])
    report["tolerance"]["formal"]["absolute_ms"] = 0.015
    with pytest.raises(ValueError, match="formal_absolute_tolerance_must_stay"):
        native_acceptance(report)
    report = _report([_item(0.1, 0.105, "a")])
    report["tolerance"] = {
        "formal": {
            "relative": 0.1, "absolute": 0.015,
            "denominator": "observed_ms",
        },
        "diagnostic": None,
        "applies_to": "paired_comparable_items_only",
    }
    with pytest.raises(ValueError, match="formal_unknown_field:absolute"):
        native_acceptance(report)
    policy = native_acceptance_policy()
    assert policy["relative_tolerance"] == 0.1
    assert policy["absolute_tolerance_ms"] == 0.0
    assert policy["formal_sample_field"] == "formal_allowed_error_ms"
    assert "historical_absolute_15us_pass" in policy["excluded_evidence"]


def test_wide_diagnostic_tolerance_does_not_change_the_formal_failure():
    diagnostic = {
        "relative": 0.5, "absolute_ms": 0.052,
        "denominator": "observed_ms",
        "application": "diagnostic_only_not_formal_acceptance",
    }
    report = _report(
        [_item(0.1, 0.13, "a", diagnostic=diagnostic)], diagnostic=diagnostic)
    result = native_acceptance(report)
    assert result["source_status"] == "failed"
    assert result["status"] == "failed"
    assert result["tolerance"]["formal"]["relative"] == 0.1
    assert result["tolerance"]["diagnostic"]["relative"] == 0.5
    sample = result["items"][0]["samples"][0]
    assert sample["within_tolerance"] is False
    assert sample["diagnostic_within"] is True
    assert result["items"][0]["status"] == "outside_tolerance"


def test_item_verdict_cannot_hide_an_outside_sample():
    report = _report([_item(0.1, 0.105, "a")])
    report["items"][0]["comparison"]["sample_verdicts"].append(
        _sample(0.09, 0.105, "observation:late"))
    with pytest.raises(ValueError, match="cannot_hide_a_formal_sample_failure"):
        native_acceptance(report)


def test_count_mismatch_is_rejected_not_silently_renormalized():
    report = _report([_item(0.1, 0.105, "a")])
    report["counts"]["paired"] = 2
    with pytest.raises(ValueError, match="count_mismatch:paired"):
        native_acceptance(report)


def test_inventory_identity_keeps_duplicate_and_capture_relations(tmp_path):
    ledger = tmp_path / "scope_map.csv"
    ledger.write_text(
        "audit_id,hardware_tier,suite,phase,in_a5_scope,primary_raw_source,"
        "event_source,capture_id,duplicate_group_id,duplicate_group_size,duplicate_role\n"
        "F1,A5,s,decode,True,/x,/x,C1,D1,2,first_reference\n"
        "F2,A5,s,decode,True,/x,/x,C1,D1,2,duplicate_reference\n"
        "F3,A5,s,decode,True,/x,/x,C2,D3,1,first_reference\n",
        encoding="utf-8")
    records = tmp_path / "records.csv"
    records.write_text(
        "hardware_tier,full_per_call_10pct_validated,per_step_runtime_state_restored,"
        "exact_per_step_values_restored,context_explicitly_unobserved,"
        "source_binding_ambiguous,configuration_recovery_status,"
        "current_evidence_kind,native_compared_cohorts\n"
        "A5,False,False,False,True,False,structured_configuration_located,"
        "partial_native_contract_cohort_mean,2\n"
        "A5,False,False,False,False,False,structured_configuration_located,"
        "family_engine_and_kernel_sum_aggregate,0\n"
        "A5,False,False,False,False,False,collection_metadata_or_scripts_located,"
        "structure_or_inventory_without_operator_comparison,0\n",
        encoding="utf-8")
    summary = tmp_path / "inventory.json"
    summary.write_text(json.dumps({
        "ledger_total_records": 3, "a5_records": 3,
        "a5_unique_event_captures": 2, "a5_duplicate_extra_records": 1,
        "a5_duplicate_groups": 1,
        "a5_records_runtime_state_not_restored": 3,
        "a5_records_context_unobserved": 1,
        "a5_records_source_binding_ambiguous": 0,
        "a5_independent_capture_counts": {"identity_check": True},
    }), encoding="utf-8")
    board = native_comparison_scoreboard(
        _report([_item(0.1, 0.105, "a")]), inventory=summary,
        ledger=ledger, records=records)
    inventory = board["inventory"]
    assert inventory["ledger"]["total_records"] == 3
    assert inventory["identity_ledger"]["duplicate_extra_records"] == 1
    assert inventory["identity_ledger"]["multi_record_duplicate_groups"] == 1
    assert inventory["identity_ledger"]["captures"]["distinct"] == 2
    assert inventory["identity_ledger"]["captures"]["records_per_capture_histogram"] \
        == {"1": 1, "2": 1}
    assert inventory["a5_record_recovery"]["flags"]["full_per_call_10pct_validated"] \
        == {"false": 3}
    assert inventory["a5_record_recovery"]["native_evidence_records"] == 1
    assert inventory["a5_record_recovery"]["native_compared_cohorts_total"] == 2
    assert board["denominators"]["inventory"]["validated_fraction_of_ledger"] is None
    summary.write_text(json.dumps({
        "ledger_total_records": 4, "a5_records": 3,
        "a5_unique_event_captures": 2, "a5_duplicate_extra_records": 1,
        "a5_duplicate_groups": 1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger_row_count_mismatch"):
        native_comparison_scoreboard(
            _report([_item(0.1, 0.105, "a")]), inventory=summary, ledger=ledger)


def test_accuracy_facade_projects_native_points_without_median_pass():
    report = _report([_item(0.1, 0.105, "a"), _item(0.1, 0.13, "b")])
    result = accuracy_report(report)
    assert result["schema"] == "mindie-sim.native-accuracy-report/1"
    assert result["status"] == "failed"
    assert result["tolerance_outcomes"]["within"] == 1
    assert result["tolerance_outcomes"]["outside"] == 1
    assert result["coverage"]["paired_items"] == 2
    assert result["coverage"]["structure_accuracy"] == "not_computed"
    points = {point["point_id"]: point for point in result["points"]}
    assert points["comparison:a"]["within_tolerance"] is True
    assert points["comparison:b"]["within_tolerance"] is False
    assert points["comparison:b"]["observation_range_ms"] == [0.1, 0.1]
    assert points["comparison:b"]["formal_allowed_error_ms"] == pytest.approx(0.01)
    assert points["comparison:b"]["samples"][0]["observed_ms"] == 0.1
    assert points["comparison:b"]["samples"][0][
        "formal_allowed_error_ms"] == pytest.approx(0.01)


def test_markdown_keeps_failure_and_7337_disclaimer():
    board = native_comparison_scoreboard(
        _report([_item(0.1, 0.105, "a"), _item(0.1, 0.13, "b")]),
        label="synthetic")
    markdown = render_native_comparison_markdown(board)
    assert "outside_tolerance" in markdown
    assert "absolute_15us" in markdown
    assert "not_7337_record_validation" in markdown
    assert "no_median_or_family_average_substitution" in markdown


def test_cli_publishes_json_and_markdown_and_returns_failed_code(tmp_path):
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(
        _report([_item(0.1, 0.105, "a"), _item(0.1, 0.13, "b")])), encoding="utf-8")
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    code = main([str(source), "--output", str(output), "--markdown", str(markdown)])
    assert code == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["denominators"]["accuracy"]["outside"] == 1
    text = markdown.read_text(encoding="utf-8")
    assert "outside_tolerance" in text and "absolute_15us" in text


def _alias_dispositions(tmp_path, captures):
    path = "/tmp/synthetic-kernel_details.csv"
    rows = []
    for index, capture in enumerate(captures, start=1):
        source_id = f"SRC-alias-{index}"
        rows.append({
            "audit_id": f"F{index}", "hardware_tier": "A5",
            "suite": "s", "phase": "prefill",
            "primary_raw_source": path, "primary_source_id": source_id,
            "source_ids": json.dumps([source_id]),
            "event_source_ids": json.dumps([source_id]),
            "evidence_capture_ids": json.dumps([capture]),
            "source_statuses": json.dumps([{
                "source_id": source_id, "event_source_id": source_id,
                "event_source": path, "capture_id": capture,
                "status": "recovered_event_abi_clock", "exists": True,
                "raw_path": path,
            }]),
            "derived": "False", "derived_key": "",
            "source_set_group_id": f"RSG-{index}",
            "source_set_group_size": "1",
            "source_set_role": "first_reference",
            "native_case_id": "",
        })
    target = tmp_path / "aliases.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def test_same_path_same_capture_alias_merges_and_different_capture_is_ambiguous(tmp_path):
    report = _report([_item(0.1, 0.105, "a")])
    acceptance = native_acceptance(report)
    merged = record_linkage(
        acceptance, dispositions=_alias_dispositions(
            tmp_path, ["CAP-alias", "CAP-alias"]))
    assert merged["status"] == "all_paired_items_linked"
    assert merged["independent_evidence"]["matched_audit_records"] == 2
    assert merged["independent_evidence"]["independent_capture_count"] == 1
    assert merged["items"][0]["status"] == "matched"
    assert merged["items"][0]["matched_audit_ids"] == ["F1", "F2"]

    ambiguous = record_linkage(
        acceptance, dispositions=_alias_dispositions(
            tmp_path, ["CAP-alias-1", "CAP-alias-2"]))
    assert ambiguous["status"] == "partial"
    assert ambiguous["comparison"]["ambiguous_items"] == 1
    assert ambiguous["comparison"]["matched_items"] == 0
    assert ambiguous["comparison"]["conflict_items"] == 0
    assert ambiguous["items"][0]["status"] == "ambiguous"
    assert "ambiguous_source_alias" in ambiguous["items"][0]["reason"]
