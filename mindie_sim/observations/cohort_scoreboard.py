"""Keep selected native-call evidence beside, but outside, model-step totals.

[Role] Summarize existing cohort comparisons for the model scoreboard.
[Invariants] No prediction is executed or adjusted. Captures, ranks, windows and
  variants are never pooled. Per-descriptor failures survive family cancellation.
  Reference rows cannot enter the compared-cohort pass denominator.
[Timeouts & Locks] Linear in a bounded supplied cohort report; no I/O or locks.
[Side-Effect Policy] Returns fresh report records; no mutation of source data.
"""
from collections import Counter
from collections.abc import Mapping
import json

from ..cohort_accuracy import accuracy_fields, accuracy_policy


def summarize_cohorts(report, *, label, provenance):
    windows = []
    for window in report["windows"]:
        groups = {}
        for cohort in window["cohorts"]:
            status = cohort["status"]
            key = (cohort["native_operator_type"], status)
            group = groups.setdefault(key, {
                "native_kind": key[0], "comparison_status": status,
                "cohorts": 0, "complete_calls": 0, "passed_cohorts": 0,
                "tier_counts": Counter(), "failures": [],
                "observed_ms": 0.0, "predicted_ms": 0.0,
                "sum_absolute_descriptor_gap_ms": 0.0,
                "unpriced_cohorts": 0,
            })
            group["cohorts"] += 1
            count = cohort["observed"]["complete_event_count"]
            group["complete_calls"] += count
            observed = cohort["observed"]["complete_activity_sum_ns"] / 1e6
            group["observed_ms"] += observed
            predicted_call = cohort.get("prediction", {}).get("per_call_ms")
            if predicted_call is None:
                group["unpriced_cohorts"] += 1
            else:
                predicted = predicted_call * count
                group["predicted_ms"] += predicted
                group["sum_absolute_descriptor_gap_ms"] += abs(predicted - observed)
            fields = accuracy_fields(cohort)
            if fields["accuracy_pass"] is not None:
                group["passed_cohorts"] += int(fields["accuracy_pass"])
                group["tier_counts"][fields["accuracy_tier"]] += 1
            if fields["accuracy_pass"] is False:
                group["failures"].append({
                    "cohort_id": cohort["cohort_id"],
                    "contract": cohort["contract"],
                    "complete_calls": count,
                    "absolute_activity_gap_ms": fields["abs_error_per_call_us"] * count / 1000,
                    **fields,
                })
        families = []
        for group in groups.values():
            group["tier_counts"] = dict(group["tier_counts"])
            group["failures"].sort(key=lambda row: -row["absolute_activity_gap_ms"])
            group["failed_cohorts"] = len(group["failures"])
            if group["unpriced_cohorts"]:
                group.update(predicted_ms=None, signed_gap_ms=None,
                             absolute_net_gap_ms=None, signed_gap_percent=None,
                             sum_absolute_descriptor_gap_ms=None)
            else:
                gap = group["predicted_ms"] - group["observed_ms"]
                group.update(signed_gap_ms=gap, absolute_net_gap_ms=abs(gap),
                             signed_gap_percent=(
                                 100 * gap / group["observed_ms"] if group["observed_ms"] else None))
            families.append(group)
        windows.append({
            "rank_id": window["rank_id"], "window": window["window"],
            "source_identity": window["source_identity"],
            "families": sorted(families, key=lambda row: -row["observed_ms"]),
        })
    return {
        "label": label, "report_source": provenance,
        "graph_id": report["graph_id"], "structure_source": report.get("structure_source", {}),
        "declared_execution": report["declared_execution"],
        "declared_kernel_bindings": report.get("declared_kernel_bindings", {}),
        "accuracy_policy": accuracy_policy(),
        "scope": "retained_native_complete_calls; isolated_predictions; not_whole_model",
        "aggregation": "separate_variant_rank_and_window; no_cross_report_sum",
        "windows": windows,
    }


def render_cohort_campaigns(campaigns):
    if not campaigns:
        return ""
    lines = [
        "\n## Selected native-call campaigns\n",
        "These retained comparisons do not change any full-step row. Each variant, "
        "rank and capture window stays separate. Activity sums use complete events; "
        "clipped rows are excluded. Family net cancellation does not hide failing "
        "descriptors. References are excluded from the pass denominator.\n",
        "| Campaign / rank / window | Native family | Status | Calls | Observed ms | Predicted ms | "
        "Signed gap ms | Sum absolute descriptor gaps ms | Per-call tiers passed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    for campaign in campaigns:
        for index, window in enumerate(campaign["windows"]):
            for family in window["families"]:
                passed = (f"{family['passed_cohorts']}/{family['cohorts']}"
                          if family["comparison_status"] == "compared" else "unscored")
                lines.append(
                    f"| {campaign['label']} / {window['rank_id']} / {index} | {family['native_kind']} | "
                    f"{family['comparison_status']} | {family['complete_calls']} | "
                    f"{number(family['observed_ms'])} | {number(family['predicted_ms'])} | "
                    f"{number(family['signed_gap_ms'])} | "
                    f"{number(family['sum_absolute_descriptor_gap_ms'])} | {passed} |")
    for campaign in campaigns:
        lines += [
            "",
            f"- {campaign['label']} report SHA256: `{campaign['report_source']['sha256']}`; "
            f"graph: `{campaign['graph_id']}`. Execution and capture identities remain in JSON.",
        ]
    return "\n".join(lines) + "\n"


def _native_cohort(item):
    identity = item.get("identity")
    boundary = item.get("boundary")
    if not isinstance(identity, Mapping) or not isinstance(boundary, Mapping):
        raise ValueError("native_acceptance_item_requires_identity_and_boundary")
    fusion = boundary.get("fusion")
    members = (fusion.get("members") if isinstance(fusion, Mapping) else None)
    if members is None:
        members = identity.get("fusion_members") or ()
    return {
        "native_operator": identity.get("native_operator"),
        "native_abi_sha256": identity.get("native_abi_sha256"),
        "boundary_kind": boundary.get("kind"),
        "hardware": identity.get("hardware"),
        "model": identity.get("model"),
        "quant": identity.get("quant"),
        "phase": identity.get("phase"),
        "implementation": identity.get("implementation"),
        "deployment_image_id": identity.get("deployment_image_id"),
        "fusion_members": sorted(str(member) for member in members),
    }


def summarize_native_comparison(acceptance):
    """Group independent native items by exact cohort identity, never by name.

    The group row is a count of individual verdicts.  It is not a mean, so a
    failing item cannot disappear inside a cohort total.
    """
    if not isinstance(acceptance, Mapping):
        raise ValueError("native_comparison_summary_requires_acceptance")
    groups = {}
    for item in acceptance.get("items", ()):
        identity = _native_cohort(item)
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, {"identity": identity, "items": []})["items"].append(item)
    rows = []
    for group in groups.values():
        items = group["items"]
        within = [item for item in items if item["status"] == "within_tolerance"]
        outside = [item for item in items if item["status"] == "outside_tolerance"]
        work_only = [item for item in items if item["status"] == "work_only"]
        timed = within + outside
        failing_samples = [
            sample.get("observation_id")
            for item in outside for sample in item["samples"]
            if not sample["within_tolerance"]
        ]
        observed = [item["observed_ms"] for item in items]
        timed_observed = [item["observed_ms"] for item in timed]
        predicted = [item["predicted_ms"] for item in timed]
        worst = (max(abs(item["relative_error"]) for item in timed)
                 if timed else None)
        rows.append({
            "cohort": group["identity"],
            "paired_items": len(items),
            "duration_denominator_items": len(timed),
            "work_only": len(work_only),
            "within": len(within),
            "outside": len(outside),
            "observation_samples": sum(len(item["samples"]) for item in timed),
            "observed_ms_range": [min(observed), max(observed)] if observed else None,
            "timed_observed_ms_range": (
                [min(timed_observed), max(timed_observed)] if timed_observed else None),
            "predicted_ms_range": [min(predicted), max(predicted)] if predicted else None,
            "worst_absolute_relative_error": worst,
            "failing_item_ids": [item["item_id"] for item in outside],
            "failing_sample_ids": [str(value) for value in failing_samples],
            "work_only_item_ids": [item["item_id"] for item in work_only],
            "work_only_reasons": sorted({
                str(item.get("comparison", {}).get("reason"))
                for item in work_only}),
            "status": (
                "contains_failures" if outside else
                "contains_work_only" if work_only and within else
                "work_only" if work_only else
                "all_items_within" if within else "empty"),
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
            "aggregation": "per_item_verdicts_not_group_mean; predicted_range_and_error_from_timed_set",
        })
    return sorted(rows, key=lambda row: (
        row["status"] != "contains_failures",
        -(row["worst_absolute_relative_error"]
          if row["worst_absolute_relative_error"] is not None else -1.0),
        row["cohort"]["native_operator"],
        row["cohort"]["native_abi_sha256"] or ""))


def render_native_comparison(rows):
    if not rows:
        return ""
    lines = [
        "\n## Independent native cohorts\n",
        "Each row counts individual same-ABI item verdicts. A cohort mean, family "
        "average or signed cancellation is not a pass criterion; failing item IDs "
        "remain below. Work-only items keep observation/work facts but contribute "
        "no duration verdict or numerical error.\n",
        "| Operator | ABI (sha256) | Boundary | Items (within/outside/work-only) | "
        "Duration denominator | Samples | Observed ms range | Predicted ms range | "
        "Worst relative error | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    def span(values):
        return (" .. ".join(number(value) for value in values)
                if values else "unavailable")

    for row in rows:
        cohort = row["cohort"]
        abi = (cohort["native_abi_sha256"] or "unavailable")
        if abi.startswith("sha256:"):
            abi = abi[7:19]
        lines.append(
            f"| {cohort['native_operator']} | {abi} | {cohort['boundary_kind']} | "
            f"{row['paired_items']} ({row['within']}/{row['outside']}/"
            f"{row['work_only']}) | "
            f"{row['duration_denominator_items']} | "
            f"{row['observation_samples']} | "
            f"{span(row['observed_ms_range'])} | "
            f"{span(row['predicted_ms_range'])} | "
            f"{number(row['worst_absolute_relative_error'])} | {row['status']} |")
    if rows and (rows[0].get("partial_coverage") or rows[0].get("truncated")):
        lines.append(
            "- Source status `" + str(rows[0].get("source_status")) + "`; "
            "`all_items_within` is local item evidence only, not a slice pass.")
    if any(row["work_only_item_ids"] for row in rows):
        lines.append(
            "- Work-only items have no predicted duration or formal error; they stay "
            "outside the duration denominator. Reasons: "
            + ", ".join(sorted({reason for row in rows
                                for reason in row["work_only_reasons"]})) + ".")
    if any(row["failing_item_ids"] for row in rows):
        lines.append("")
    for row in rows:
        if row["failing_item_ids"]:
            lines.append(f"- {row['cohort']['native_operator']} failing items: "
                         + ", ".join(row["failing_item_ids"]))
    return "\n".join(lines) + "\n"
