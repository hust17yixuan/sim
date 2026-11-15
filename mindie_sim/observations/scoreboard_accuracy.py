"""Project the shared per-call criterion onto retained scoreboard boundaries.

[Role] Add conditional diagnostic tiers and independent whole-step acceptance.
[Invariants] A reference never becomes a scored cohort. Per-call fields require
  retained counts. Mixed descriptors are tiered separately, never by pooled mean.
  No criterion changes predicted time or removes cumulative residuals.
[Timeouts & Locks] Linear work in finite report rows; no locks or waits.
[Side-Effect Policy] Pure copies of supplied report mappings; no I/O or devices.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from ..cohort_accuracy import (
    NATIVE_ACCEPTANCE_POLICY, accuracy_fields, accuracy_policy,
    native_acceptance, native_acceptance_policy,
)
from ..coverage import (
    comparison_coverage, inventory_coverage, record_linkage,
)
from .cohort_scoreboard import render_native_comparison, summarize_native_comparison


CONDITIONAL_STATUSES = {
    "conditional", "conditional_reference", "conditional_closed_call_blocks",
    "conditional_semantic_stage_projection", "conditional_source_phase",
}


def step_fields(pair):
    """The whole step always uses 15%, including steps shorter than 50 us."""
    result = {
        "step_accuracy_tolerance": {"unit": "ratio", "value": 0.15},
        "step_accuracy_scope": "conditional_whole_step_not_heldout_validation",
        "step_accuracy_pass": None,
        "step_accuracy_status": "unscored",
    }
    if pair["status"] == "conditional_range":
        endpoints = [step_fields(row) for row in pair["scenarios"]]
        result["step_accuracy_endpoints"] = endpoints
        if endpoints and all(row["step_accuracy_pass"] is not None for row in endpoints):
            result["step_accuracy_pass"] = all(row["step_accuracy_pass"] for row in endpoints)
    elif pair["status"] == "conditional" and pair.get("absolute_gap_percent") is not None:
        result["step_accuracy_pass"] = pair["absolute_gap_percent"] <= 15
    if result["step_accuracy_pass"] is not None:
        result["step_accuracy_status"] = (
            "within_relative_tolerance" if result["step_accuracy_pass"] else "outside_tolerance")
    return result


def call_fields(row, *, homogeneous=False):
    """Retain per-call arithmetic, with a separate descriptive-only criterion."""
    count = row.get("occurrences")
    has_count = isinstance(count, (int, float)) and math.isfinite(count) and count > 0
    paired = row.get("status") in CONDITIONAL_STATUSES
    observed = row.get("observed_mean_ms") if has_count else None
    predicted = row.get("predicted_weighted_mean_ms", row.get("predicted_ms")) if has_count else None
    if has_count and observed is None and row.get("observed_sum_ms") is not None:
        observed = row["observed_sum_ms"] / count
    if has_count and predicted is None and row.get("predicted_sum_ms") is not None and paired:
        predicted = row["predicted_sum_ms"] / count
    if not paired:
        predicted = None
    record = {
        "status": "reference_only" if paired else "unscored",
        "observed": {"duration_ns": {"mean": None if observed is None else observed * 1e6}},
        "prediction": {"per_call_ms": predicted},
    }
    fields = accuracy_fields(record)
    fields["accuracy_scope"] = "conditional_diagnostic_excluded_from_scored_cohort_denominator"
    fields["descriptive_tier"] = None
    if not has_count:
        fields["per_call_ineligible_reason"] = "retained_call_count_unavailable"
    elif not paired or predicted is None or observed is None:
        fields["per_call_ineligible_reason"] = "matching_complete_call_prediction_unavailable"
    elif not homogeneous:
        fields["per_call_ineligible_reason"] = "mixed_or_unconfirmed_descriptor_population"
    elif observed <= 0:
        fields["per_call_ineligible_reason"] = "positive_complete_event_mean_unavailable"
    else:
        fields["descriptive_tier"] = {
            **accuracy_fields({**record, "status": "compared"}),
            "scope": "conditional_numeric_diagnostic_not_scored_cohort_acceptance",
        }
    return fields


def annotate_call(row, *, homogeneous=None):
    if homogeneous is None:
        homogeneous = row.get("descriptor_groups") == 1 or row.get("closed_block_comparison") is True
    return {**deepcopy(row), **call_fields(row, homogeneous=homogeneous)}


def group_fields(row):
    """Only a homogeneous matched group supports a descriptive per-call tier."""
    if row.get("comparison_level") == "coarse_semantic_family":
        return {
            **call_fields({"status": row["status"]}),
            "per_call_ineligible_reason": "native_and_program_grouping_not_equivalent",
        }
    left, right = row.get("observed") or {}, row.get("predicted") or {}
    same_count = (left.get("occurrences") == right.get("occurrences")
                  and left.get("occurrences") is not None)
    counts = left["occurrences"] if same_count else None
    summed = (row.get("comparison_level") == "semantic_family"
              or left.get("statistic") == right.get("statistic") == "sum")
    mean = left.get("statistic") == right.get("statistic") == "mean"
    values = {
        "status": row["status"], "occurrences": counts if summed or mean else None,
        "observed_sum_ms": left.get("duration_ms") if summed else None,
        "predicted_sum_ms": right.get("duration_ms") if summed else None,
        "observed_mean_ms": left.get("duration_ms") if mean else None,
        "predicted_weighted_mean_ms": right.get("duration_ms") if mean else None,
    }
    return call_fields(values, homogeneous=(
        row.get("comparison_level") != "semantic_family" and bool(left.get("identity"))))


def policy():
    return {
        "per_call": accuracy_policy(),
        "whole_step": {
            "relative_tolerance": 0.15,
            "scope": "conditional_whole_step_not_heldout_validation",
        },
        "conditional_references": (
            "Per-call numeric diagnostics are separate from scored cohort acceptance; "
            "accuracy_pass remains null. Mixed descriptors are not tiered by pooled mean."),
        "small_call_tolerance": (
            "An acceptance allowance, not measured launch-overhead attribution or a timing floor."),
    }


def annotate_kimi_report(original):
    """Refresh saved descriptor reports without re-executing their predictions."""
    report = deepcopy(original)
    report["accuracy_policy"] = policy()
    report["rows"] = [annotate_call(row, homogeneous=True) for row in report["rows"]]
    for field in ("family_totals", "grouped_reference_domains"):
        for total in report.get(field, ()):
            members = [row for row in report["rows"]
                       if (row.get("family") == total.get("family") if field == "family_totals"
                           else row.get("matrix_groups") == total.get("matrix_groups"))]
            total.update(call_fields(total, homogeneous=len(members) == 1))
            tiers = [row["descriptive_tier"] for row in members if row["descriptive_tier"]]
            residuals = [row["sum_error"] for row in members if row.get("sum_error") is not None]
            total["descriptor_diagnostics"] = {
                "scope": "descriptive_only_not_scored_cohort_denominator",
                "descriptor_count": len(members),
                "tiered_descriptor_count": len(tiers),
                "untiered_descriptor_count": len(members) - len(tiers),
                "statuses": dict(Counter(row["accuracy_status"] for row in tiers)),
                "tiers": dict(Counter(row["accuracy_tier"] for row in tiers)),
                "residual_descriptor_count": len(residuals),
                "signed_cumulative_gap_ms": (
                    math.fsum(row["signed_gap_ms"] for row in residuals) if residuals else None),
                "absolute_descriptor_cumulative_gap_ms": (
                    math.fsum(row["absolute_gap_ms"] for row in residuals) if residuals else None),
            }
    return report


def render_kimi_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    lines = [
        "# Kimi descriptor-conditioned criterion", "",
        "Conditional numeric diagnostics only; these references do not enter scored "
        "cohort acceptance. Observed means above 50 us use relative 15%; means at or "
        "below 50 us use absolute 15 us. No predicted duration is changed.", "",
        "| Family | Calls | Observed cumulative ms | Predicted cumulative ms | Signed / absolute descriptor gap ms | Descriptor tier outcomes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["family_totals"]:
        diagnostic = row["descriptor_diagnostics"]
        lines.append(
            f"| {row['family']} | {row['occurrences']} | {number(row['observed_sum_ms'])} | "
            f"{number(row.get('predicted_sum_ms'))} | "
            f"{number((row.get('sum_error') or {}).get('signed_gap_ms'))} / "
            f"{number(row.get('descriptor_absolute_residual_sum_ms'))} | "
            f"{diagnostic['statuses']}; untiered {diagnostic['untiered_descriptor_count']} |")
    lines.extend([
        "", "Family means do not select a tier for mixed descriptors. Cumulative activity "
        "is separate from a whole-model step; descriptor residuals retain cancellation.", "",
        "| Descriptor | Family | Calls | Observed us/call | Predicted us/call | Signed / absolute us/call | Conditional tier outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["rows"]:
        tier = row["descriptive_tier"]
        outcome = tier["accuracy_status"] if tier else row.get("per_call_ineligible_reason", "unscored")
        lines.append(
            f"| {row['id']} | {row['family']} | {row.get('occurrences', 'unavailable')} | "
            f"{number(row['observed_per_call_us'])} | {number(row['predicted_per_call_us'])} | "
            f"{number(row['per_call_diff_us'])} / {number(row['abs_error_per_call_us'])} | {outcome} |")
    return "\n".join(lines) + "\n"


NATIVE_SCOREBOARD_SCHEMA = "mindie-sim.native-comparison-scoreboard/1"


def native_comparison_scoreboard(comparison, *, inventory=None, captures=None,
                                 ledger=None, records=None, captures_manifest=None,
                                 record_dispositions=None, record_sources=None,
                                 source_registry=None, captures_index=None,
                                 source_edges=None, label=None, source=None):
    """Project one profiler native comparison into the public sim slice.

    The downstream report reuses the comparison owner's paired items.  It never
    re-prices, medians a failure away, or counts an unpaired side as correct.
    """
    acceptance = native_acceptance(comparison)
    linkage_inputs = (record_dispositions, record_sources, source_registry,
                      captures_index, captures_manifest)
    if inventory is None and any(value is not None for value in (
            captures, ledger, records, *linkage_inputs)):
        raise ValueError("native_scoreboard_inventory_required_for_capture_inputs")
    if record_dispositions is None and any(value is not None for value in (
            record_sources, source_registry, captures_index, source_edges)):
        raise ValueError("native_scoreboard_record_dispositions_required_for_linkage")
    inventory_report = None
    coverage = None
    linkage = None
    if inventory is not None:
        inventory_report = inventory_coverage(
            inventory, captures, ledger=ledger, records=records,
            captures_manifest=captures_manifest)
        if record_dispositions is not None:
            linkage = record_linkage(
                acceptance, dispositions=record_dispositions,
                record_sources=record_sources,
                source_registry=source_registry,
                captures_index=captures_index,
                captures_manifest=captures_manifest,
                source_edges=source_edges)
        coverage = comparison_coverage(
            inventory_report, acceptance, linkage=linkage)
    counts = acceptance["counts"]
    policy = native_acceptance_policy()
    report = {
        "schema": NATIVE_SCOREBOARD_SCHEMA,
        "status": acceptance["status"],
        "label": label,
        "source_schema": acceptance["source_schema"],
        "source_status": acceptance["source_status"],
        "source": source,
        "source_inputs": deepcopy(comparison.get("inputs")),
        "source_boundary_policy": deepcopy(acceptance["boundary_policy"]),
        "source_interpretation": acceptance["source_interpretation"],
        "policy": policy,
        "tolerance": acceptance["tolerance"],
        "coverage": deepcopy(acceptance["coverage"]),
        "coverage_reconciled": acceptance["coverage_reconciled"],
        "subset_scope": deepcopy(acceptance["subset_scope"]),
        "parent_capture": acceptance["parent_capture"],
        "partial_coverage": acceptance["partial_coverage"],
        "truncated": acceptance["truncated"],
        "window": deepcopy(acceptance["window"]),
        "windows": deepcopy(acceptance["windows"]),
        "source_identity": deepcopy(acceptance["source_identity"]),
        "observation_counts": deepcopy(acceptance["observation_counts"]),
        "legacy_evidence": {
            "accepted_as_this_round": [NATIVE_ACCEPTANCE_POLICY],
            "rejected_as_this_round": list(policy["excluded_evidence"]),
            "historical_absolute_15us_pass_accepted": False,
            "in_sample_or_mean_reproduction_accepted": False,
            "training_sample_resplit_used_as_generalization": False,
            "diagnostic_tolerance_used_as_formal": False,
            "work_only_counted_as_measured_duration": False,
            "source_status_authoritative": True,
        },
        "denominators": {
            "accuracy": {
                "basis": "paired_comparable_native_items",
                "paired_items": counts["paired_items"],
                "duration_denominator_items": counts["duration_denominator_items"],
                "duration_denominator_basis": (
                    "within_tolerance+outside_tolerance_items_with_real_predicted_duration"),
                "work_only": counts["work_only"],
                "within": counts["within"],
                "outside": counts["outside"],
                "observation_samples": counts["observation_samples"],
                "condition_gap_items_in_denominator": counts[
                    "condition_gap_items_in_denominator"],
                "unpaired_observations": counts["unpaired_observations"],
                "unpaired_predictions": counts["unpaired_predictions"],
                "ambiguous": counts["ambiguous"],
                "source_status": acceptance["source_status"],
                "partial_coverage": acceptance["partial_coverage"],
                "truncated": acceptance["truncated"],
                "excluded_from_duration_denominator": {
                    "work_only_items": counts["work_only"],
                    "unpaired_observations": counts["unpaired_observations"],
                    "unpaired_predictions": counts["unpaired_predictions"],
                    "ambiguous": counts["ambiguous"],
                },
            },
            "inventory": coverage,
        },
        "counts": acceptance["source_counts"],
        "cohorts": summarize_native_comparison(acceptance),
        "items": acceptance["items"],
        "unpaired_observations": acceptance["unpaired_observations"],
        "unpaired_predictions": acceptance["unpaired_predictions"],
        "ambiguous": acceptance["ambiguous"],
        "failure_matrix": acceptance["failure_matrix"],
        "inventory": inventory_report,
        "record_linkage": linkage,
        "consumer_contract": {
            "schema": NATIVE_SCOREBOARD_SCHEMA,
            "status": acceptance["status"],
            "source_status": acceptance["source_status"],
            "accuracy_denominator": (
                "denominators.accuracy.duration_denominator_items"),
            "accuracy_counts": "denominators.accuracy",
            "duration_denominator": (
                "denominators.accuracy.duration_denominator_items "
                "(within_tolerance+outside_tolerance items with real predicted duration)"),
            "paired_items": (
                "denominators.accuracy.paired_items (all ABI/boundary pairs, "
                "including work_only)"),
            "work_only": (
                "denominators.accuracy.work_only and items[].status=work_only; "
                "no predicted duration, numerical error or within/outside verdict"),
            "precision_denominator": (
                "record_linkage.independent_evidence.precision_denominator_paired_items "
                "equals denominators.accuracy.duration_denominator_items"),
            "items": "items[].formal_allowed_error_ms and items[].samples[]",
            "unpaired": "unpaired_observations/unpaired_predictions/ambiguous",
            "failure_matrix": "failure_matrix",
            "record_linkage": "record_linkage",
            "inventory": "inventory",
            "coverage": "coverage/coverage_reconciled/subset_scope/parent_capture",
        },
        "structure_accuracy": {
            "computed": False,
            "reason": "Unknown, unpaired or ambiguous structural items never enter a "
                      "correctness denominator.",
        },
        "non_claims": list(acceptance["non_claims"]) + [
            "independent_native_slice_only_not_7337_record_validation",
            "no_historical_absolute_15us_or_in_sample_mean_pass",
            "no_training_sample_resplit_as_generalization",
            "no_median_or_family_average_substitution",
            "no_work_only_item_in_duration_denominator",
            "no_unpaired_or_ambiguous_item_in_accuracy_denominator",
            "no_item_only_pass_when_source_is_partial_truncated_or_inconclusive",
            "no_diagnostic_tolerance_as_formal_acceptance",
            "record_match_is_identity_linkage_not_per_call_certification",
            "duplicate_or_derived_records_do_not_increase_accuracy_denominator",
        ],
        "interpretation": list(acceptance["interpretation"]) + [
            "The reported status describes this slice only; a complete candidate still "
            "requires T19 full validation.",
        ],
    }
    return report


def render_native_comparison_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    accuracy = report["denominators"]["accuracy"]
    lines = [
        "# Independent native comparison",
        "",
        f"Slice status: `{report['status']}`; source comparison status: "
        f"`{report['source_status']}`.",
        "",
        "Criterion: `tolerance.formal` gives `abs(predicted - observed) / observed "
        "<= 10%` for every item and every retained sample that has a real predicted "
        "duration. A diagnostic tolerance is retained for diagnosis only and never "
        "changes the formal verdict. There is no absolute-error exemption, no median "
        "substitution and no family mean. Work-only items retain observation/work "
        "facts but have no duration verdict or numerical error. A historical "
        "`absolute_15us` or in-sample mean reproduction is not accepted here.",
        "",
        "## Accuracy denominator",
        "",
        "| Paired ABI items | Duration denominator | Within | Outside | Work only | "
        "Samples | Condition gaps | Unpaired observations | Unpaired predictions | "
        "Ambiguous |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {accuracy['paired_items']} | {accuracy['duration_denominator_items']} | "
        f"{accuracy['within']} | {accuracy['outside']} | {accuracy['work_only']} | "
        f"{accuracy['observation_samples']} | "
        f"{accuracy['condition_gap_items_in_denominator']} | "
        f"{accuracy['unpaired_observations']} | {accuracy['unpaired_predictions']} | "
        f"{accuracy['ambiguous']} |",
        "",
        "`paired_items` is the ABI/boundary pair set (includes work-only items). "
        "`duration_denominator_items` is `within + outside`: only items with a real "
        "predicted duration and formal verdict. Work-only items are not passes and "
        "not failures. Unpaired and ambiguous sides stay below and never become "
        "correct by being unknown.",
    ]
    window = report.get("window") or {}
    source_identity = report.get("source_identity") or {}
    window_kinds = sorted({str(row.get("kind"))
                           for row in (report.get("windows") or ())})
    lines += [
        "",
        "## Source coverage",
        "",
        f"- Source status `{report['source_status']}`; partial_coverage="
        f"`{report['partial_coverage']}`; truncated=`{report['truncated']}`; "
        f"window kind `{window.get('kind', 'unavailable')}`; "
        f"window count `{window.get('count', 'unavailable')}`; "
        f"window kinds `{window_kinds}`.",
        f"- Source identity: `{source_identity.get('path', 'unavailable')}`; "
        f"sha256 `{source_identity.get('sha256', 'unavailable')}`; "
        f"scope `{source_identity.get('sha256_scope', 'unavailable')}`.",
        "- Local item verdicts are kept even when the source window is incomplete; "
        "that does not make the slice passed.",
    ]
    diagnostic = (report.get("tolerance") or {}).get("diagnostic")
    if diagnostic:
        lines.append(
            "- Diagnostic tolerance retained separately: "
            f"relative={diagnostic.get('relative')}, "
            f"absolute_ms={diagnostic.get('absolute_ms')}; it never changes the "
            "formal verdict.")
    inventory = report.get("inventory")
    if inventory:
        ledger = inventory["ledger"]
        identity = inventory.get("identity_ledger") or {}
        recovery = inventory.get("a5_record_recovery") or {}
        flags = recovery.get("flags", {})
        validated = flags.get("full_per_call_10pct_validated") or {}
        validated_text = (", ".join(f"{key}:{value}" for key, value in validated.items())
                          if validated else "unavailable")
        lines += [
            "",
            "## 7337-record inventory identity",
            "",
            "The historical ledger is carried as record identity, not as independent "
            "captures and not as this slice's accuracy denominator.",
            "",
            "| Ledger records | A5 records | Captures (manifest) | A5 captures | "
            "Source-set duplicate extras | Legacy duplicate extras |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {ledger['total_records']} | {ledger.get('a5_scope_records')} | "
            f"{ledger.get('captures_total')} | {ledger.get('a5_capture_count')} | "
            f"{ledger.get('source_set_duplicate_extra_records')} | "
            f"{ledger.get('legacy_a5_duplicate_extra_records') if ledger.get('legacy_a5_duplicate_extra_records') is not None else 'n/a'} |",
            "",
            f"- Capture count source: `{ledger.get('capture_count_source')}`; actual "
            f"manifest version retained in `inventory.capture_manifest`.",
            f"- Legacy identity-ledger duplicate groups: "
            f"{identity.get('duplicate_groups', 'unavailable')}; multi-record groups: "
            f"{identity.get('multi_record_duplicate_groups', 'unavailable')}.",
            f"- A5 runtime state not restored: "
            f"{report['inventory'].get('unrecovered', {}).get('runtime_state_not_restored_records')}; "
            f"A5 current 10% per-call validated flags: {validated_text}.",
            f"- Comparison coverage: {report['denominators']['inventory']['statement']}",
        ]
    linkage = report.get("record_linkage")
    if linkage:
        independent = linkage["independent_evidence"]
        lines += [
            "",
            "## Ledger record linkage",
            "",
            "Record matching is identity/coverage linkage, not per-call certification. "
            "Duplicate source-set references and derived rows remain visible but do "
            "not increase the paired-item precision denominator.",
            "",
            "| Link status | Paired ABI items | Duration denominator | Work only | "
            "Matched items | Unmatched | Conflict | Matched audit records | "
            "Independent captures | Duplicate refs | Derived | Precision denominator |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |",
            f"| {linkage['status']} | {linkage['comparison']['paired_items']} | "
            f"{linkage['comparison']['duration_denominator_items']} | "
            f"{linkage['comparison']['work_only_items']} | "
            f"{linkage['comparison']['matched_items']} | "
            f"{linkage['comparison']['unmatched_items']} | "
            f"{linkage['comparison']['conflict_items']} | "
            f"{independent['matched_audit_records']} | "
            f"{independent['independent_capture_count']} | "
            f"{independent['duplicate_reference_records']} | "
            f"{independent['derived_records']} | "
            f"{independent['precision_denominator_paired_items']} |",
        ]
        for row in linkage["items"]:
            if row["status"] != "matched":
                lines.append(
                    f"- {row['item_id']}: {row['status']} ({row.get('reason')}); "
                    f"matched records {row['matched_record_count']}.")
    if report["partial_coverage"] or report["truncated"]:
        lines.append(
            "- Cohort row `all_items_within` is local item evidence only; the overall "
            "slice remains incomplete because the source window is partial/truncated.")
    cohort_text = render_native_comparison(report.get("cohorts", ()))
    if cohort_text:
        lines.extend(cohort_text.splitlines())
    lines += [
        "",
        "## Per-item results",
        "",
        "| Item | Operator | Boundary | Observed ms | Predicted ms | Allowed ms | "
        "Relative error | Samples within/total | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report["items"]:
        samples = item.get("samples", ())
        within = sum(sample["within_tolerance"] for sample in samples)
        lines.append(
            f"| {item['item_id']} | {item['identity'].get('native_operator')} | "
            f"{item['boundary'].get('kind')} | {number(item['observed_ms'])} | "
            f"{number(item.get('predicted_ms'))} | "
            f"{number(item.get('formal_allowed_error_ms'))} | "
            f"{number(item.get('relative_error'))} | {within}/{len(samples)} | "
            f"{item['status']} |")
    work_only_items = [item for item in report["items"]
                       if item["status"] == "work_only"]
    if work_only_items:
        lines += [
            "",
            "## Work-only items (no duration verdict)",
            "",
            "These ABI pairs retain the real observation and physical work facts. "
            "No predicted duration, formal allowed error or within/outside result is "
            "invented, so they are excluded from the duration denominator.",
            "",
            "| Item | Operator | Observed ms | Cube MACs | Issued MACs | Tensor bytes | Reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in work_only_items:
            work = item.get("work") or {}
            cube = work.get("cube") or {}
            byte_values = work.get("bytes") or {}
            lines.append(
                f"| {item['item_id']} | {item['identity'].get('native_operator')} | "
                f"{number(item['observed_ms'])} | {number(cube.get('macs'))} | "
                f"{number(cube.get('issued_macs'))} | "
                f"{number(byte_values.get('total'))} | "
                f"{item.get('comparison', {}).get('reason', 'unavailable')} |")
    if report["unpaired_observations"] or report["unpaired_predictions"] \
            or report["ambiguous"]:
        lines += [
            "",
            "## Unpaired and ambiguous sides",
            "",
            "| Side | Identity / ids | Value ms | Reason |",
            "| --- | --- | ---: | --- |",
        ]
        for row in report["unpaired_observations"]:
            lines.append(
                f"| observation | {row.get('observation_ids') or row.get('request_id')} | "
                f"{number(row.get('observed_ms'))} | {row.get('reason')} |")
        for row in report["unpaired_predictions"]:
            lines.append(
                f"| prediction | {row.get('prediction_id')} | "
                f"{number(row.get('predicted_ms'))} | {row.get('reason')} |")
        for row in report["ambiguous"]:
            lines.append(
                f"| ambiguous | {row.get('request_id')} | unavailable | "
                f"{row.get('reason')} |")
    if report["failure_matrix"]:
        lines += [
            "",
            "## Failure matrix",
            "",
            "| Root cause | Reason | Count | Item ids | Claim |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for row in report["failure_matrix"]:
            item_ids = ", ".join(row.get("item_ids", ())[:5])
            lines.append(
                f"| {row.get('root_cause')} | {row.get('reason')} | "
                f"{row.get('count')} | {item_ids} | {row.get('claim')} |")
    lines += ["", "## Non-claims", ""]
    lines += [f"- {value}" for value in report["non_claims"]]
    return "\n".join(lines) + "\n"
