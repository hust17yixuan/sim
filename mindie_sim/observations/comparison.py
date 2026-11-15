"""Descriptive same-boundary residuals beside independent accuracy validation.

[Role] Compare explicitly identified predictions with retained observations and
  project native-ABI residuals without certifying accuracy.
[Invariants] Descriptive matches never promote sample splits or certify accuracy.
  Native pairing needs the content-addressed ABI and boundary; unknown semantics
  remain unmatched and both unpaired sides survive missing data.
[Timeouts & Locks] One observation pass; retain matching scalar durations only.
[Side-Effect Policy] Pure report construction; no calibration or source writes.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from statistics import median

from .accuracy import _duration, _identity_key, accuracy_report


def comparison_report(campaign):
    """Keep all-split descriptive residuals distinct from heldout validation.

    Observations may be an iterator. An explicit prediction identity is the
    only join key; this function never finds a neighboring shape or recipe.
    """
    predictions = list(campaign["predictions"])
    selected = {_identity_key(point["identity"])[0] for point in predictions}
    groups, seen, boundaries = {}, set(), set()

    def collect():
        for sample in campaign.get("observations", ()):
            key, absent = _identity_key(sample["identity"])
            boundaries.add(sample["identity"].get("boundary"))
            if key in selected and not absent and sample["sample_id"] not in seen:
                seen.add(sample["sample_id"])
                group = groups.setdefault(key, {
                    "runs": defaultdict(list), "splits": Counter(), "source_preview": [],
                    "unresolved_semantics": Counter(),
                })
                unresolved = sample.get("metadata", {}).get("unresolved_semantics", ())
                if unresolved:
                    group["unresolved_semantics"].update(unresolved)
                else:
                    _duration(sample["duration_ms"], observation=True)
                    group["runs"][sample["run_id"]].append(sample["duration_ms"])
                    group["splits"][sample["split"]] += 1
                    if len(group["source_preview"]) < 5:
                        group["source_preview"].append({
                            "sample_id": sample["sample_id"], "source": sample["source"],
                            "split": sample["split"],
                        })
            yield sample

    validation = accuracy_report({**campaign, "predictions": predictions, "observations": collect()})
    points = []
    for prediction in predictions:
        key, absent = _identity_key(prediction["identity"])
        group = groups.get(key)
        point = {
            "point_id": prediction["point_id"], "identity": prediction["identity"],
            "predicted_ms": prediction["duration_ms"], "status": "unpaired",
            "comparison": None,
        }
        points.append(point)
        if absent:
            point.update(reason="incomplete_identity", missing_identity_fields=absent)
        elif not group:
            point["reason"] = "no_exact_observation_identity"
        elif not group["runs"]:
            point.update(reason="unresolved_observation_semantics",
                         unresolved_semantics=dict(group["unresolved_semantics"]))
        else:
            medians = {run: median(values) for run, values in group["runs"].items()}
            observed = median(medians.values())
            residual = prediction["duration_ms"] - observed
            point.update(
                status="descriptive_match", observed_ms=observed,
                statistic="median_of_run_medians",
                acquisition_runs=len(medians), samples=sum(group["splits"].values()),
                sample_splits=dict(group["splits"]), run_medians_ms=medians,
                observation_preview=group["source_preview"],
                unresolved_semantics=dict(group["unresolved_semantics"]),
                comparison={
                    "signed_residual_ms": residual, "absolute_residual_ms": abs(residual),
                    "relative_error": residual / observed,
                },
                interpretation="descriptive_all_split_comparison_not_independent_validation",
            )
    return {
        "schema": "mindie-sim.comparison-report/1",
        "descriptive": {
            "matched_points": sum(point["status"] == "descriptive_match" for point in points),
            "unpaired_points": sum(point["status"] == "unpaired" for point in points),
            "available_boundaries": sorted(value for value in boundaries if value is not None),
            "points": points,
            "interpretation": (
                "Descriptive residuals include original unassigned/training observations. "
                "They have no tolerance or acceptance result and do not establish generalization."
            ),
        },
        "validation": validation,
    }


NATIVE_IDENTITY_FIELDS = (
    "native_operator", "native_abi_sha256", "boundary",
)
NATIVE_CONDITION_FIELDS = (
    "hardware", "phase", "model", "quant", "deployment_image_id",
    "implementation", "execution", "attrs", "api", "version",
)
_NATIVE_STRUCTURAL_FIELDS = (
    "native_operator", "native_abi_sha256", "boundary", "scope", "expansion",
    "rank", "device_id", "stream", "clock_domain", "window_id",
    "fusion_members",
)


def _native_conditions(identity):
    """Return the full declared condition mapping, including new keys/null.

    New campaign records carry ``identity.conditions``.  The legacy top-level
    fields are accepted as a fallback and are merged only when the complete
    mapping does not already declare that key, so an explicit null survives.
    """
    raw = identity.get("conditions")
    conditions = dict(raw) if isinstance(raw, Mapping) else {}
    # Legacy callers express a condition at top level; an explicit top-level
    # value (including null) is the caller's declared fact and overrides the
    # frozen mapping rather than being silently ignored.
    for key in NATIVE_CONDITION_FIELDS:
        if key in identity:
            conditions[key] = identity[key]
    return conditions


def _native_flat_conditions(value):
    flat = {}

    def visit(item, path):
        if isinstance(item, Mapping):
            if not item:
                flat[path or "conditions"] = {}
                return
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        flat[path or "conditions"] = item

    visit(dict(value or {}), "")
    return flat


def _native_identity(record):
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        return None, ["identity"]
    raw_boundary = identity.get("boundary")
    boundary = raw_boundary.get("kind") if isinstance(raw_boundary, Mapping) else raw_boundary
    fusion_members = identity.get("fusion_members")
    if fusion_members is None and isinstance(raw_boundary, Mapping):
        fusion = raw_boundary.get("fusion")
        fusion_members = fusion.get("members") if isinstance(fusion, Mapping) else ()
    conditions = _native_conditions(identity)
    values = {
        "native_operator": identity.get("native_operator"),
        "native_abi_sha256": identity.get("native_abi_sha256"),
        "boundary": boundary,
        "scope": identity.get("scope"),
        "expansion": identity.get("expansion"),
        "rank": identity.get("rank"),
        "device_id": identity.get("device_id"),
        "stream": identity.get("stream"),
        "clock_domain": identity.get("clock_domain"),
        "window_id": identity.get("window_id"),
        "conditions": conditions,
        "fusion_members": tuple(sorted(fusion_members or ())),
    }
    for key in NATIVE_CONDITION_FIELDS:
        values[key] = conditions.get(key)
    missing = [field for field in NATIVE_IDENTITY_FIELDS
               if not values.get(field)]
    if not conditions.get("hardware"):
        missing.append("hardware")
    if missing:
        return None, missing
    return values, []


def _native_key(identity):
    conditions = _native_conditions(identity)
    payload = {
        key: identity.get(key) for key in _NATIVE_STRUCTURAL_FIELDS
    }
    payload["conditions"] = conditions
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _native_identity_differences(left, right):
    differences = []
    for field in _NATIVE_STRUCTURAL_FIELDS:
        if left.get(field) != right.get(field):
            differences.append(field)
    left_flat = _native_flat_conditions(_native_conditions(left))
    right_flat = _native_flat_conditions(_native_conditions(right))
    for path in sorted(set(left_flat) | set(right_flat)):
        if (path not in left_flat or path not in right_flat
                or left_flat[path] != right_flat[path]):
            name = path.split(".", 1)[0]
            if name not in differences:
                differences.append(name)
    return sorted(differences)


def _native_observation_detail(group):
    durations = [sample["duration_ms"] for sample in group["samples"]]
    return {
        "identity": group["identity"],
        "request_ids": sorted(group["request_ids"]),
        "observation_ids": [sample["sample_id"] for sample in group["samples"]],
        "observed_ms": median(durations),
        "samples": len(group["samples"]),
        "observation_range_ms": [min(durations), max(durations)],
    }


def native_comparison_report(campaign):
    """Descriptive same-boundary native residuals with both unpaired sides.

    This report never assigns a tolerance verdict.  Pairing requires the exact
    content-addressed native ABI, operator, boundary, rank/stream and declared
    conditions.  A kernel or API name is not a key, and a fused observation is
    not split among its semantic members.
    """
    observations = list(campaign.get("observations", ()))
    predictions = list(campaign.get("predictions", ()))
    groups, missing_observations, seen_samples = {}, [], {}
    for sample in observations:
        identity, missing = _native_identity(sample)
        if missing:
            missing_observations.append({
                "sample_id": sample.get("sample_id"), "reason": "incomplete_identity",
                "missing_identity_fields": missing,
            })
            continue
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("unresolved_semantics"):
            missing_observations.append({
                "sample_id": sample.get("sample_id"),
                "reason": "unresolved_observation_semantics",
            })
            continue
        _duration(sample["duration_ms"], observation=True)
        key = _native_key(identity)
        sample_id = sample.get("sample_id")
        signature = (key, sample["duration_ms"], sample.get("run_id"),
                     sample.get("source"))
        if sample_id in seen_samples:
            if seen_samples[sample_id] != signature:
                missing_observations.append({
                    "sample_id": sample_id,
                    "reason": "conflicting_native_observation",
                })
            continue
        seen_samples[sample_id] = signature
        group = groups.setdefault(key, {
            "identity": identity, "samples": [],
            "request_ids": set(), "fusion_members": identity["fusion_members"],
        })
        group["samples"].append(sample)
        if sample.get("request_id"):
            group["request_ids"].add(sample["request_id"])
    by_request = {}
    for key, group in groups.items():
        for request_id in group["request_ids"]:
            by_request.setdefault(request_id, set()).add(key)
    ambiguous_requests = {
        request_id for request_id, keys in by_request.items() if len(keys) > 1
    }
    groups_by_key = {}
    for key, group in groups.items():
        groups_by_key.setdefault(key, []).append(group)
    predictions_by_key, predictions_by_request, unpaired_predictions = {}, {}, []
    for prediction in predictions:
        identity, missing = _native_identity(prediction)
        request_id = prediction.get("request_id")
        if missing:
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": "incomplete_identity",
                "missing_identity_fields": missing,
            })
            continue
        status = str(prediction.get("status") or "modeled")
        if status not in {"modeled", "predicted", "ok"}:
            reason = prediction.get("reason") or prediction.get("unmodeled_reason")
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": f"prediction_{status}" + (f":{reason}" if reason else ""),
            })
            continue
        try:
            duration = _duration(prediction["duration_ms"])
        except (KeyError, TypeError, ValueError):
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": "invalid_prediction_duration",
            })
            continue
        row = {
            "prediction_id": prediction.get("prediction_id"),
            "request_id": request_id, "identity": identity,
            "duration_ms": duration,
            "evidence": list(prediction.get("evidence") or ()),
        }
        if request_id:
            predictions_by_request.setdefault(request_id, []).append(row)
        else:
            predictions_by_key.setdefault(_native_key(identity), []).append(row)
    points, not_comparable, used = [], [], set()
    paired_keys, not_comparable_keys = set(), set()
    for key, group in groups.items():
        identity = group["identity"]
        if group["request_ids"] & ambiguous_requests:
            not_comparable.append({
                "reason": "ambiguous_request_identity",
                **_native_observation_detail(group),
            })
            not_comparable_keys.add(key)
            continue
        by_id = [
            row for request_id in group["request_ids"]
            for row in predictions_by_request.get(request_id, ())
        ]
        if not by_id and len(groups_by_key.get(key, [])) > 1:
            not_comparable.append({
                "reason": "ambiguous_observation_candidates",
                **_native_observation_detail(group),
                "candidate_request_ids": [
                    candidate["request_ids"] for candidate in groups_by_key[key]
                ],
            })
            not_comparable_keys.add(key)
            continue
        candidates = by_id or predictions_by_key.get(key, [])
        if not candidates:
            continue
        exact = [row for row in candidates
                 if _native_key(row["identity"]) == key]
        if not exact:
            differences = {
                row["prediction_id"]: _native_identity_differences(
                    identity, row["identity"])
                for row in candidates
            }
            fields = sorted({
                field for values in differences.values() for field in values})
            reason = ("fusion_boundary_mismatch" if fields == ["fusion_members"]
                      else "request_identity_conflict:" + ",".join(fields))
            not_comparable.append({
                "reason": reason,
                **_native_observation_detail(group),
                "prediction_ids": [row["prediction_id"] for row in candidates],
                "identity_differences": differences,
            })
            not_comparable_keys.add(key)
            continue
        candidates = exact
        if len(candidates) > 1:
            not_comparable.append({
                "reason": "ambiguous_predictions",
                **_native_observation_detail(group),
                "prediction_ids": [row["prediction_id"] for row in candidates],
            })
            not_comparable_keys.add(key)
            used.update(row["prediction_id"] for row in candidates)
            continue
        prediction = candidates[0]
        used.add(prediction["prediction_id"])
        observed = median(sample["duration_ms"] for sample in group["samples"])
        residual = prediction["duration_ms"] - observed
        sample_differences = []
        for sample in group["samples"]:
            sample_residual = prediction["duration_ms"] - sample["duration_ms"]
            sample_differences.append({
                "sample_id": sample["sample_id"],
                "observed_ms": sample["duration_ms"],
                "signed_residual_ms": sample_residual,
                "absolute_residual_ms": abs(sample_residual),
                "relative_error": sample_residual / sample["duration_ms"],
                "denominator": "observed_ms",
            })
        points.append({
            "point_id": "native-point:" + _native_key(identity)[:32],
            "request_id": prediction.get("request_id"),
            "identity": identity,
            "prediction_id": prediction["prediction_id"],
            "observed_ms": observed,
            "predicted_ms": prediction["duration_ms"],
            "statistic": "median_of_observations",
            "samples": len(group["samples"]),
            "runs": len({sample.get("run_id") for sample in group["samples"]}),
            "observation_range_ms": [
                min(sample["duration_ms"] for sample in group["samples"]),
                max(sample["duration_ms"] for sample in group["samples"]),
            ],
            "comparison": {
                "signed_residual_ms": residual,
                "absolute_residual_ms": abs(residual),
                "relative_error": residual / observed,
                "denominator": "observed_ms",
                "sample_differences": sample_differences,
                "worst_absolute_relative_error": max(
                    abs(row["relative_error"]) for row in sample_differences),
            },
            "prediction_evidence": prediction["evidence"],
            "interpretation": "descriptive_same_boundary_difference_not_acceptance",
        })
        paired_keys.add(key)
    unpaired_observations = []
    for key, group in groups.items():
        if key in paired_keys or key in not_comparable_keys:
            continue
        unpaired_observations.append({
            "request_ids": sorted(group["request_ids"]),
            "identity": group["identity"],
            "observation_ids": [sample["sample_id"] for sample in group["samples"]],
            "observed_ms": median(sample["duration_ms"] for sample in group["samples"]),
            "samples": len(group["samples"]),
            "reason": "no_exact_prediction_identity",
        })
    for rows in (*predictions_by_key.values(), *predictions_by_request.values()):
        for row in rows:
            if row["prediction_id"] not in used:
                unpaired_predictions.append({
                    "prediction_id": row["prediction_id"],
                    "request_id": row.get("request_id"),
                    "identity": row["identity"],
                    "predicted_ms": row["duration_ms"],
                    "reason": "no_exact_observation_identity",
                })
    status = ("completed" if points and not not_comparable and not unpaired_observations
              and not unpaired_predictions and not missing_observations
              else "partial")
    return {
        "schema": "mindie-sim.native-comparison/1",
        "status": status,
        "cohort_role": campaign.get("cohort_role", "unspecified"),
        "fixed_boundary": {
            "join": "native_operator+native_abi_sha256+boundary+rank/stream+conditions",
            "denominator": "observed_ms",
            "name_only_allowed": False,
            "family_average_allowed": False,
            "fusion_member_allocation_allowed": False,
        },
        "coverage": {
            "observations": len(observations),
            "predictions": len(predictions),
            "paired": len(points),
            "not_comparable": len(not_comparable),
            "not_comparable_observation_samples": sum(
                int(entry.get("samples", 0)) for entry in not_comparable),
            "unpaired_observations": len(unpaired_observations),
            "unpaired_predictions": len(unpaired_predictions),
            "excluded_observations": len(missing_observations),
        },
        "points": points,
        "not_comparable": not_comparable,
        "unpaired_observations": unpaired_observations,
        "unpaired_predictions": unpaired_predictions,
        "excluded_observations": missing_observations,
        "non_claims": [
            "descriptive_residual_not_acceptance",
            "no_name_only_pairing",
            "no_family_average_pass",
            "no_fused_member_allocation",
        ],
        "interpretation": [
            "Residuals are descriptive same-boundary differences, not acceptance.",
            "A name-only or family-level match is never emitted.",
            "A fused observation retains its semantic members and is not allocated.",
            "Unpaired and unresolved sides remain visible.",
        ],
    }
