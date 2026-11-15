"""Public structural coverage separate from prediction accuracy.

[Role] Carry configuration omissions to every public prediction boundary.
[Invariants] Partial costs describe only modeled components, never an accuracy
  claim. Missing measurements neither change coverage nor prevent execution.
[Timeouts & Locks] Constant-size metadata reads; no I/O or locks.
[Side-Effect Policy] Returns a new JSON-ready mapping.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
from hashlib import sha256
import json
import os
from pathlib import Path


MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_FILE_BYTES = 64 * 1024 * 1024
MAX_INVENTORY_ROWS = 100_000
INVENTORY_SCHEMA = "mindie-sim.inventory-coverage/1"
COMPARISON_COVERAGE_SCHEMA = "mindie-sim.comparison-coverage/1"


def model_coverage(model):
    metadata = dict(model.metadata)
    coverage = metadata.get("model_coverage", "complete")
    return {
        "model_coverage": coverage,
        "unmodeled_features": list(metadata.get("unmodeled_features", ())),
        "degraded_submodules": list(metadata.get("degraded_submodules", ())),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if coverage == "partial" else "declared_model_structure"),
    }


def combine_coverage(models):
    reports = [model_coverage(model) for model in models]
    partial = any(report["model_coverage"] == "partial" for report in reports)
    return {
        "model_coverage": "partial" if partial else "complete",
        "unmodeled_features": sorted({gap for report in reports for gap in report["unmodeled_features"]}),
        "degraded_submodules": sorted({gap for report in reports for gap in report["degraded_submodules"]}),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if partial else "declared_model_structure"),
    }


def source_identity(value):
    """Hash one bounded real input; the report keeps its concrete identity."""
    path = Path(value)
    size = path.stat().st_size
    if size > MAX_IDENTITY_FILE_BYTES:
        raise ValueError("inventory_source_exceeds_64_mib")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": size}


def _read_object(value, name, max_bytes=MAX_METADATA_BYTES):
    source = source_identity(value) if isinstance(value, (str, Path)) else None
    if source is not None:
        if source["size_bytes"] > max_bytes:
            raise ValueError(f"{name}_exceeds_16_mib")
        value = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_requires_object")
    return value, source


def _int_field(container, key, *, required=False):
    value = container.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"inventory_field_requires_nonnegative_integer:{key}")
    return value


def _ledger_identity(value, expected_rows):
    source = source_identity(value)
    if source["size_bytes"] > MAX_IDENTITY_FILE_BYTES:
        raise ValueError("inventory_ledger_exceeds_64_mib")
    digest = sha256()
    rows = 0
    tiers, scopes, roles, suites, phases = (Counter() for _ in range(5))
    group_counts, capture_counts, declared_sizes = Counter(), Counter(), {}
    a5_group_counts, a5_capture_counts = Counter(), Counter()
    a5_rows = 0
    with Path(value).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        base = {"audit_id", "hardware_tier", "suite", "phase", "capture_id"}
        new_schema = {"source_set_group_id", "source_set_group_size",
                      "source_set_role"} <= fields
        old_schema = {"duplicate_group_id", "duplicate_group_size",
                      "duplicate_role"} <= fields
        if not base <= fields or not (new_schema or old_schema):
            raise ValueError("inventory_ledger_missing_identity_columns")
        group_key = "source_set_group_id" if new_schema else "duplicate_group_id"
        size_key = "source_set_group_size" if new_schema else "duplicate_group_size"
        role_key = "source_set_role" if new_schema else "duplicate_role"
        scope_key = "in_a5_scope" if "in_a5_scope" in fields else None
        digest_fields = ["audit_id", "hardware_tier", "suite", "phase", "capture_id",
                         group_key, size_key, role_key]
        for row in reader:
            rows += 1
            if rows > MAX_INVENTORY_ROWS:
                raise ValueError("inventory_ledger_exceeds_100000_rows")
            digest.update("|".join(
                str(row.get(key, "")) for key in digest_fields).encode("utf-8") + b"\n")
            tier = row.get("hardware_tier", "")
            tiers[tier] += 1
            if scope_key is not None:
                in_a5 = str(row.get(scope_key, "")).strip().lower() in {
                    "true", "1", "yes"}
                scopes[row[scope_key]] += 1
            else:
                in_a5 = tier == "A5"
                scopes["A5" if in_a5 else "other"] += 1
            roles[row.get(role_key, "")] += 1
            suites[row.get("suite", "")] += 1
            phases[row.get("phase", "")] += 1
            a5_rows += int(in_a5)
            group = row.get(group_key, "")
            if group:
                group_counts[group] += 1
                if in_a5:
                    a5_group_counts[group] += 1
                try:
                    declared = int(row.get(size_key) or 1)
                except (TypeError, ValueError):
                    raise ValueError(
                        "inventory_ledger_duplicate_group_size_not_integer") from None
                if declared_sizes.setdefault(group, declared) != declared:
                    raise ValueError("inventory_ledger_duplicate_group_size_conflict")
            capture_id = row.get("capture_id", "")
            if capture_id:
                capture_counts[capture_id] += 1
                if in_a5:
                    a5_capture_counts[capture_id] += 1
    if rows != expected_rows:
        raise ValueError("inventory_ledger_row_count_mismatch")
    for group, count in group_counts.items():
        if declared_sizes.get(group) != count:
            raise ValueError("inventory_ledger_duplicate_group_size_mismatch")
    histogram = Counter(capture_counts.values())
    a5_histogram = Counter(a5_capture_counts.values())
    return {
        "source": source,
        "schema_family": "source_set" if new_schema else "duplicate_group",
        "rows": rows,
        "a5_rows": a5_rows,
        "a5_duplicate_groups": sum(
            1 for count in a5_group_counts.values() if count > 1),
        "a5_duplicate_extra_records": sum(
            count - 1 for count in a5_group_counts.values() if count > 1),
        "a5_captures": {
            "distinct": len(a5_capture_counts),
            "records_per_capture_histogram": {
                str(size): count for size, count in sorted(a5_histogram.items())},
        },
        "in_a5_scope_counts": dict(sorted(scopes.items())),
        "hardware_tier_counts": dict(sorted(tiers.items())),
        "suite_counts": dict(sorted(suites.items())),
        "phase_counts": dict(sorted(phases.items())),
        "duplicate_role_counts": dict(sorted(roles.items())),
        "duplicate_groups": len(group_counts),
        "multi_record_duplicate_groups": sum(
            1 for count in group_counts.values() if count > 1),
        "duplicate_extra_records": sum(
            count - 1 for count in group_counts.values()),
        "captures": {
            "distinct": len(capture_counts),
            "records_per_capture_histogram": {
                str(size): count for size, count in sorted(histogram.items())},
        },
        "identity_sha256_in_file_order": "sha256:" + digest.hexdigest(),
        "identity_fields": digest_fields,
    }


def _a5_record_recovery(value, expected_rows):
    source = source_identity(value)
    if source["size_bytes"] > MAX_IDENTITY_FILE_BYTES:
        raise ValueError("inventory_records_exceed_64_mib")
    flag_keys = (
        "full_per_call_10pct_validated", "per_step_runtime_state_restored",
        "exact_per_step_values_restored", "context_explicitly_unobserved",
        "source_binding_ambiguous")
    flags = {key: Counter() for key in flag_keys}
    present = set()
    configuration, evidence = Counter(), Counter()
    native_records = native_cohorts = 0
    total_rows = rows = 0
    with Path(value).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if "hardware_tier" not in fields:
            raise ValueError("inventory_records_missing_hardware_tier")
        present = {key for key in flag_keys if key in fields}
        if "configuration_recovery_status" in fields:
            present.add("configuration_recovery_status")
        evidence_key = ("current_evidence_kind" if "current_evidence_kind" in fields
                        else "record_grain" if "record_grain" in fields else None)
        for row in reader:
            total_rows += 1
            if total_rows > MAX_INVENTORY_ROWS:
                raise ValueError("inventory_records_exceed_100000_rows")
            if str(row.get("hardware_tier", "")).strip() != "A5":
                continue
            rows += 1
            for key in present & set(flag_keys):
                raw = str(row.get(key, "")).strip().lower()
                if raw in {"true", "1", "yes"}:
                    value = "true"
                elif raw in {"false", "0", "no"}:
                    value = "false"
                else:
                    raise ValueError(f"inventory_records_non_boolean_field:{key}")
                flags[key][value] += 1
            if "configuration_recovery_status" in present:
                configuration[row.get("configuration_recovery_status", "")] += 1
            if evidence_key:
                evidence[row.get(evidence_key, "")] += 1
            try:
                count = int(row.get("native_compared_cohorts") or 0)
            except ValueError:
                raise ValueError("inventory_records_native_compared_cohorts_not_integer") from None
            if count < 0:
                raise ValueError("inventory_records_native_compared_cohorts_negative")
            native_records += int(count > 0)
            native_cohorts += count
    if rows != expected_rows:
        raise ValueError("inventory_records_a5_row_count_mismatch")
    flag_counts = {}
    for key in flag_keys:
        if key in present:
            flag_counts[key] = dict(sorted(flags[key].items()))
        else:
            flag_counts[key] = {"not_declared": rows}
    return {
        "source": source,
        "source_rows_total": total_rows,
        "rows": rows,
        "record_scope": "A5",
        "flags": flag_counts,
        "native_evidence_records": native_records,
        "native_compared_cohorts_total": native_cohorts,
        "configuration_recovery_status_counts": dict(sorted(configuration.items())),
        "evidence_kind_counts": dict(sorted(evidence.items())),
        "historical_tier_note": (
            "This ledger does not supply a current 10% per-call verdict; "
            "a historical absolute_15us flag is not consumed here."),
    }


def _csv_records(value, name, required):
    source = source_identity(value)
    if source["size_bytes"] > MAX_IDENTITY_FILE_BYTES:
        raise ValueError(f"{name}_exceeds_64_mib")
    rows = []
    with Path(value).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or ()):
            raise ValueError(f"{name}_missing_required_columns")
        for row in reader:
            rows.append(row)
            if len(rows) > MAX_INVENTORY_ROWS:
                raise ValueError(f"{name}_exceeds_100000_rows")
    return rows, source


def _json_list(value, name):
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}_must_be_json_list") from None
    if not isinstance(parsed, list):
        raise ValueError(f"{name}_must_be_json_list")
    return parsed


def _path_key(value):
    return os.path.realpath(str(value)) if value else ""


def _edge_list(row, name):
    values = _json_list(row.get("source_statuses"), name)
    edges, seen = [], set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name}_requires_edge_objects")
        source_id = value.get("source_id")
        capture_id = value.get("capture_id")
        event_source = value.get("event_source")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{name}_edge_requires_source_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError(f"{name}_edge_requires_capture_id")
        key = (source_id, capture_id, _path_key(event_source or value.get("raw_path")))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source_id": source_id,
            "event_source_id": value.get("event_source_id"),
            "capture_id": capture_id,
            "event_source": event_source,
            "raw_path": value.get("raw_path"),
            "status": value.get("status"),
            "exists": value.get("exists"),
        })
    if not edges:
        raise ValueError(f"{name}_requires_source_status_edges")
    return edges


def _declared_source_edges(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        value = value.get("source_edges", value)
    if isinstance(value, Mapping):
        rows = [dict(edge, item_id=item_id)
                for item_id, edges in value.items()
                for edge in (edges if isinstance(edges, list) else [edges])]
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise ValueError("source_edges_requires_object_or_list")
    by_item = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source_edge_requires_object")
        item_id = (row.get("item_id") or row.get("sample_id")
                   or row.get("observation_id") or row.get("window_id"))
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("source_edge_requires_item_id")
        source_id = row.get("source_id")
        source_path = row.get("source_path")
        capture_id = row.get("capture_id")
        if not source_id and not source_path:
            raise ValueError("source_edge_requires_source_id_or_path")
        by_item.setdefault(item_id, []).append({
            "source_id": source_id,
            "source_path": _path_key(source_path) if source_path else None,
            "capture_id": capture_id,
        })
    return by_item


def record_linkage(acceptance, *, dispositions, record_sources=None,
                   source_registry=None, captures_index=None,
                   captures_manifest=None, source_edges=None):
    """Link T16 sample/window/request facts to actual T03 source edges.

    A record may declare several sources and a source may be shared by direct
    and derived records.  Only the source edge actually selected by the
    comparison participates in capture/audit accounting; the complete source
    set stays for inventory traceability.  This is identity linkage, not
    per-call certification.
    """
    if not isinstance(acceptance, Mapping):
        raise ValueError("record_linkage_requires_acceptance")
    disposition_rows, disposition_source = _csv_records(
        dispositions, "record_dispositions",
        {"audit_id", "hardware_tier", "suite", "phase", "primary_raw_source",
         "primary_source_id", "source_ids", "event_source_ids",
         "evidence_capture_ids", "source_statuses", "derived", "derived_key",
         "source_set_group_id",
         "source_set_group_size", "source_set_role", "native_case_id"})
    records = {}
    for row in disposition_rows:
        audit_id = row.get("audit_id")
        if not audit_id or audit_id in records:
            raise ValueError("record_linkage_duplicate_or_missing_audit_id")
        edges = _edge_list(row, f"record_statuses[{audit_id}]")
        capture_ids = _json_list(row.get("evidence_capture_ids"),
                                 "record_capture_ids")
        edge_captures = {edge["capture_id"] for edge in edges}
        declared_captures = {str(value) for value in capture_ids}
        conflict = None
        if declared_captures != edge_captures:
            conflict = "evidence_capture_ids_mismatch"
        by_source = {}
        for edge in edges:
            by_source.setdefault(edge["source_id"], set()).add(edge["capture_id"])
        if any(len(values) > 1 for values in by_source.values()):
            conflict = conflict or "same_source_conflicting_capture"
        records[audit_id] = {
            "audit_id": audit_id,
            "hardware_tier": row.get("hardware_tier"),
            "suite": row.get("suite"),
            "phase": row.get("phase"),
            "record_grain": row.get("record_grain"),
            "primary_raw_source": row.get("primary_raw_source"),
            "primary_source_id": row.get("primary_source_id") or None,
            "source_ids": _json_list(row.get("source_ids"), "record_source_ids"),
            "event_source_ids": _json_list(
                row.get("event_source_ids"), "record_event_source_ids"),
            "capture_ids": capture_ids,
            "source_edges": edges,
            "edges_by_source": by_source,
            "edge_conflict": conflict,
            "derived": str(row.get("derived", "")).strip().lower() in {
                "true", "1", "yes"},
            "derived_key": row.get("derived_key") or None,
            "source_set_group_id": row.get("source_set_group_id") or None,
            "source_set_group_size": int(row.get("source_set_group_size") or 1),
            "source_set_role": row.get("source_set_role") or None,
            "native_case_id": row.get("native_case_id") or None,
        }

    sources = {"record_dispositions": disposition_source}
    path_to_sources = {}
    for audit_id, record in records.items():
        for edge in record["source_edges"]:
            source_id = edge["source_id"]
            for value in (edge["event_source"], edge["raw_path"]):
                path = _path_key(value)
                if path:
                    path_to_sources.setdefault(path, set()).add(source_id)
        primary_path = _path_key(record.get("primary_raw_source"))
        if primary_path:
            primary_source_id = record.get("primary_source_id")
            if primary_source_id and primary_source_id in record["edges_by_source"]:
                path_to_sources.setdefault(primary_path, set()).add(primary_source_id)
            elif not primary_source_id:
                matches = {edge["source_id"] for edge in record["source_edges"]
                           if _path_key(edge["event_source"]) == primary_path
                           or _path_key(edge["raw_path"]) == primary_path}
                if len(matches) == 1:
                    only = next(iter(matches))
                    path_to_sources.setdefault(primary_path, set()).add(only)

    audit_paths = {}
    if record_sources is not None:
        rows, source = _csv_records(
            record_sources, "record_sources",
            {"audit_id", "source_id", "raw_path", "canonical_path",
             "event_source"})
        sources["record_sources"] = source
        for row in rows:
            audit_id = row["audit_id"]
            if audit_id not in records:
                continue
            for key in (row.get("canonical_path"), row.get("event_source"),
                        row.get("raw_path")):
                path = _path_key(key)
                if path:
                    audit_paths.setdefault(audit_id, set()).add(path)
                    source_id = row.get("source_id")
                    if source_id:
                        path_to_sources.setdefault(path, set()).add(source_id)
    if source_registry is not None:
        rows, source = _csv_records(
            source_registry, "source_registry",
            {"source_id", "canonical_path", "event_sources", "record_ids"})
        sources["source_registry"] = source
        for row in rows:
            source_id = row["source_id"]
            paths = {_path_key(row.get("canonical_path"))}
            paths.update(_path_key(value) for value in _json_list(
                row.get("event_sources"), "source_event_sources"))
            for path in paths:
                if path:
                    path_to_sources.setdefault(path, set()).add(source_id)

    capture_by_id, capture_by_path, path_to_captures = {}, {}, {}
    if captures_index is not None:
        rows, source = _csv_records(
            captures_index, "captures_index",
            {"capture_id", "event_source", "tiers", "record_count"})
        sources["captures_index"] = source
        for row in rows:
            capture_id = row["capture_id"]
            if capture_id in capture_by_id:
                raise ValueError("record_linkage_duplicate_capture_id")
            capture_by_id[capture_id] = {
                "capture_id": capture_id,
                "event_source": row.get("event_source"),
                "tiers": _json_list(row.get("tiers"), "capture_tiers"),
                "record_count": int(row.get("record_count") or 0),
            }
            path = _path_key(row.get("event_source"))
            if path:
                capture_by_path[path] = capture_id
                path_to_captures.setdefault(path, set()).add(capture_id)
    manifest = None
    if captures_manifest is not None:
        manifest, manifest_source = _read_object(
            captures_manifest, "captures_manifest")
        manifest_count = _int_field(manifest, "capture_count", required=True)
        if capture_by_id and len(capture_by_id) != manifest_count:
            raise ValueError("record_linkage_capture_index_manifest_mismatch")
        manifest = {"capture_count": manifest_count, "source": manifest_source}
        sources["captures_manifest"] = manifest_source

    source_records, source_captures = {}, {}
    for audit_id, record in records.items():
        for edge in record["source_edges"]:
            source_records.setdefault(edge["source_id"], set()).add(audit_id)
            source_captures.setdefault(edge["source_id"], set()).add(
                edge["capture_id"])

    def resolve_reference(reference):
        source_id = reference.get("source_id")
        source_path = reference.get("source_path")
        capture_id = reference.get("capture_id")
        candidates = set()
        if source_id:
            candidates.add(source_id)
        elif source_path:
            candidates.update(path_to_sources.get(source_path, ()))
            if not candidates and source_path in capture_by_path:
                capture = capture_by_path[source_path]
                candidates.update(
                    sid for sid, captures in source_captures.items()
                    if capture in captures)
        else:
            return [], "source_edge_unresolved"
        if capture_id:
            filtered = {sid for sid in candidates
                        if capture_id in source_captures.get(sid, set())}
            if not filtered:
                return sorted(candidates), "declared_capture_conflicts_with_source"
            candidates = filtered
        if not candidates:
            return [], "source_not_in_ledger"
        if len(candidates) > 1:
            capture_sets = {
                tuple(sorted(source_captures.get(sid, ())))
                for sid in candidates}
            if len(capture_sets) != 1:
                return sorted(candidates), "ambiguous_source_alias"
        return sorted(candidates), None

    raw_declared = _declared_source_edges(source_edges)
    item_ids = {item.get("item_id") for item in acceptance.get("items", ())}
    observation_to_item, window_to_item = {}, {}
    for item in acceptance.get("items", ()):
        for observation_id in item.get("observation_ids") or ():
            observation_to_item[observation_id] = item.get("item_id")
        for window_id in item.get("observation_window_ids") or ():
            window_to_item[window_id] = item.get("item_id")
        for sample in item.get("samples", ()):
            if sample.get("observation_id"):
                observation_to_item[sample["observation_id"]] = item.get("item_id")
            if sample.get("window_id"):
                window_to_item[sample["window_id"]] = item.get("item_id")
    declared_edges = {}
    for key, edges in raw_declared.items():
        target = key
        if target not in item_ids:
            target = observation_to_item.get(key) or window_to_item.get(key)
        if target is None:
            raise ValueError(f"source_edge_unknown_item_reference:{key}")
        declared_edges.setdefault(target, []).extend(edges)

    def item_actual_edges(item):
        item_id = item.get("item_id")
        declared = declared_edges.get(item_id, ())
        references = []
        if declared:
            references = list(declared)
        else:
            identity = acceptance.get("source_identity") or {}
            source_path = _path_key(identity.get("path"))
            if not source_path:
                return [], ["no_source_identity"], False
            references = [{"source_path": source_path}]
        raw, errors, ambiguous = [], [], False
        for reference in references:
            source_ids, error = resolve_reference(reference)
            if error == "ambiguous_source_alias":
                ambiguous = True
                errors.append(error)
            elif error:
                errors.append(error)
            for source_id in source_ids:
                raw.append((source_id, reference.get("capture_id")))
        return raw, errors, ambiguous

    items = []
    matched_audits, matched_captures, matched_source_sets = set(), set(), set()
    duplicate_records, derived_records, native_cases = set(), set(), set()
    for item in acceptance.get("items", ()):
        source_path = _path_key(
            (acceptance.get("source_identity") or {}).get("path"))
        sample_ids = list(item.get("observation_ids") or ())
        window_ids = list(item.get("observation_window_ids") or ())
        for sample in item.get("samples", ()):
            if sample.get("window_id") and sample["window_id"] not in window_ids:
                window_ids.append(sample["window_id"])
        if not sample_ids:
            sample_ids = [sample.get("observation_id")
                          for sample in item.get("samples", ())
                          if sample.get("observation_id")]
        row = {
            "item_id": item.get("item_id"),
            "request_id": item.get("request_id"),
            "prediction_id": item.get("prediction_id"),
            "source_path": source_path or None,
            "source_capture_id": None,
            "observation_ids": sample_ids,
            "observation_window_ids": window_ids,
            "observation_samples": len(sample_ids),
            "selected_source_edges": [],
            "candidate_audit_ids": [],
            "candidate_capture_ids": [],
            "candidate_source_set_ids": [],
            "matched_audit_ids": [],
            "matched_record_count": 0,
            "independent_capture_count": 0,
            "independent_source_set_count": 0,
            "duplicate_reference_audit_ids": [],
            "derived_audit_ids": [],
            "native_case_ids": [],
            "status": "unmatched",
            "reason": None,
        }
        actual, reasons, ambiguous = item_actual_edges(item)
        if ambiguous:
            row.update(status="ambiguous",
                       reason=",".join(sorted(set(reasons))) or "ambiguous_source_alias",
                       source_candidates=[value[0] for value in actual])
            items.append(row)
            continue
        if not actual:
            row["reason"] = reasons[0] if reasons else "no_source_identity"
            if "declared_capture_conflicts_with_source" in reasons:
                row["status"] = "conflict"
            items.append(row)
            continue
        selected_audits, selected_captures, selected_conflicts = (
            set(), set(), list(reasons))
        selected_edges = []
        for source_id, expected_capture in actual:
            captures = source_captures.get(source_id, set())
            if not captures:
                selected_conflicts.append("source_edge_has_no_capture")
                continue
            if len(captures) > 1:
                selected_conflicts.append("same_source_multiple_captures")
            capture = sorted(captures)[0]
            if expected_capture and expected_capture not in captures:
                selected_conflicts.append("declared_capture_conflicts_with_source")
            for audit_id in source_records.get(source_id, ()):
                record = records[audit_id]
                if record["edge_conflict"]:
                    selected_conflicts.append(
                        f"record_capture_declaration_conflict:{audit_id}")
                    continue
                if capture not in record["edges_by_source"].get(source_id, set()):
                    selected_conflicts.append(
                        f"record_missing_actual_source_edge:{audit_id}")
                    continue
                selected_audits.add(audit_id)
                selected_captures.add(capture)
            selected_edges.append({"source_id": source_id, "capture_id": capture})
        if not selected_edges and not selected_audits:
            row["reason"] = (selected_conflicts[0] if selected_conflicts
                             else "source_not_in_ledger")
            items.append(row)
            continue
        selected = [records[audit_id] for audit_id in sorted(selected_audits)]
        source_sets = sorted({record["source_set_group_id"]
                              for record in selected if record["source_set_group_id"]})
        phases = sorted({record["phase"] for record in selected if record["phase"]})
        identity_phase = item.get("identity", {}).get("phase")
        if identity_phase and phases and identity_phase not in phases:
            selected_conflicts.append("phase_conflict")
        candidate_audits = [record["audit_id"] for record in selected]
        row.update({
            "selected_source_edges": selected_edges,
            "candidate_audit_ids": candidate_audits,
            "candidate_capture_ids": sorted(selected_captures),
            "candidate_source_set_ids": source_sets,
            "phase_values": phases,
        })
        if selected_conflicts:
            row.update(status="conflict",
                       reason=",".join(sorted(set(selected_conflicts))))
        else:
            row.update({
                "status": "matched",
                "matched_audit_ids": candidate_audits,
                "matched_record_count": len(selected),
                "independent_capture_count": len(selected_captures),
                "independent_source_set_count": len(source_sets),
                "duplicate_reference_audit_ids": [
                    record["audit_id"] for record in selected
                    if record["source_set_role"]
                    and record["source_set_role"] != "first_reference"],
                "derived_audit_ids": [record["audit_id"] for record in selected
                                      if record["derived"]],
                "native_case_ids": sorted({
                    record["native_case_id"] for record in selected
                    if record["native_case_id"]}),
            })
            matched_audits.update(row["matched_audit_ids"])
            matched_captures.update(selected_captures)
            matched_source_sets.update(source_sets)
            duplicate_records.update(row["duplicate_reference_audit_ids"])
            derived_records.update(row["derived_audit_ids"])
            native_cases.update(row["native_case_ids"])
        items.append(row)

    paired = int((acceptance.get("counts") or {}).get("paired_items") or 0)
    counts = acceptance.get("counts") or {}
    within = int(counts.get("within") or 0)
    outside = int(counts.get("outside") or 0)
    duration_denominator = int(
        counts.get("duration_denominator_items", within + outside) or 0)
    work_only = int(counts.get("work_only") or 0)
    matched_items = sum(row["status"] == "matched" for row in items)
    unmatched_items = sum(row["status"] == "unmatched" for row in items)
    conflict_items = sum(row["status"] == "conflict" for row in items)
    ambiguous_items = sum(row["status"] == "ambiguous" for row in items)
    status = ("all_paired_items_linked"
              if items and matched_items == len(items) == paired
              else "empty" if not items else "partial")
    return {
        "schema": "mindie-sim.record-linkage/1",
        "status": status,
        "ledger": {
            "records": len(records),
            "a5_records": sum(1 for record in records.values()
                              if record["hardware_tier"] == "A5"),
            "captures_manifest": manifest,
            "captures_index": len(capture_by_id) if capture_by_id else None,
        },
        "comparison": {
            "paired_items": paired,
            "duration_denominator_items": duration_denominator,
            "work_only_items": work_only,
            "matched_items": matched_items,
            "unmatched_items": unmatched_items,
            "conflict_items": conflict_items,
            "ambiguous_items": ambiguous_items,
            "observation_samples": sum(row["observation_samples"] for row in items),
            "matched_observation_samples": sum(
                row["observation_samples"] for row in items
                if row["status"] == "matched"),
        },
        "independent_evidence": {
            "precision_denominator_paired_items": duration_denominator,
            "duration_denominator_items": duration_denominator,
            "paired_items_retained": paired,
            "work_only_items_retained": work_only,
            "record_match_changes_precision_denominator": False,
            "matched_audit_records": len(matched_audits),
            "independent_capture_count": len(matched_captures),
            "independent_source_set_count": len(matched_source_sets),
            "duplicate_reference_records": len(duplicate_records),
            "derived_records": len(derived_records),
            "native_case_ids": sorted(native_cases),
            "statement": (
                "Ledger record matching follows the actual source/capture edge "
                "selected by the comparison. It is identity/coverage linkage, not "
                "per-call certification. Complete source sets and shared sources "
                "are inventory facts; duplicate source-set references and derived "
                "rows do not increase the paired-item precision denominator. The "
                "precision denominator is the real duration denominator "
                "(within+outside); work-only ABI pairs remain associated but add "
                "no duration score."),
        },
        "items": items,
        "unmatched_item_ids": [row["item_id"] for row in items
                               if row["status"] == "unmatched"],
        "conflict_item_ids": [row["item_id"] for row in items
                              if row["status"] == "conflict"],
        "ambiguous_item_ids": [row["item_id"] for row in items
                               if row["status"] == "ambiguous"],
        "sources": sources,
    }


def inventory_coverage(summary, captures=None, *, ledger=None, records=None,
                       captures_manifest=None):
    """Carry the 7337-record ledger identity, duplicate and capture relations.

    Accepts both the r1 summary shape and T03 r2 ``t03/full-recovery/2``.
    Capture counts come from the supplied manifest/summary version, never a
    compiled-in 4173.
    """
    summary, summary_source = _read_object(summary, "inventory_summary")
    total = _int_field(summary, "ledger_total_records")
    if total is None:
        total = _int_field(summary, "records_total", required=True)
    a5_records = _int_field(summary, "a5_records")
    legacy_captures = _int_field(summary, "a5_unique_event_captures")
    legacy_duplicate_extra = _int_field(summary, "a5_duplicate_extra_records")
    legacy_duplicate_groups = _int_field(summary, "a5_duplicate_groups")
    if a5_records is not None and a5_records > total:
        raise ValueError("inventory_a5_records_exceed_ledger_total")
    if a5_records is not None and legacy_captures is not None \
            and legacy_duplicate_extra is not None \
            and legacy_captures + legacy_duplicate_extra != a5_records:
        raise ValueError("inventory_a5_capture_duplicate_counts_do_not_reconcile")
    identity_counts = summary.get("a5_independent_capture_counts") or {}
    if not isinstance(identity_counts, Mapping):
        raise ValueError("inventory_a5_independent_capture_counts_requires_object")
    native_summary = summary.get("native") if isinstance(summary.get("native"), Mapping) else {}
    native_case_count = _int_field(summary, "a5_native_case_count")
    if native_case_count is None:
        native_case_count = native_summary.get("archive_cases")
    native_cohort_records = _int_field(summary, "a5_native_cohort_records")
    if native_cohort_records is None:
        native_cohort_records = native_summary.get("validation_rows")

    capture_report = capture_source = None
    if captures is not None:
        capture_report, capture_source = _read_object(captures, "captures_summary")
    captures_total = _int_field(summary, "captures_total")
    if capture_report is not None and _int_field(capture_report, "capture_count") is not None:
        captures_total = _int_field(capture_report, "capture_count")
    capture_count_source = ("captures_summary" if capture_report is not None
                            else "inventory_summary")
    a5_capture_count = None
    if capture_report is not None:
        tier_counts = capture_report.get("tier_capture_counts")
        if isinstance(tier_counts, Mapping):
            value = tier_counts.get("A5")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                a5_capture_count = value
    manifest_report = manifest_source = None
    if captures_manifest is not None:
        manifest_report, manifest_source = _read_object(
            captures_manifest, "captures_manifest")
        manifest_count = _int_field(manifest_report, "capture_count", required=True)
        manifest_lines = _int_field(manifest_report, "total_lines")
        if manifest_lines is not None and manifest_lines != manifest_count:
            raise ValueError("inventory_capture_manifest_total_lines_mismatch")
        if captures_total is not None and captures_total != manifest_count:
            raise ValueError("inventory_capture_manifest_count_mismatch")
        captures_total = manifest_count
        capture_count_source = "captures_manifest"

    result = {
        "schema": INVENTORY_SCHEMA,
        "source_schema": summary.get("schema"),
        "ledger": {
            "total_records": total,
            "records_are_not_independent_captures": True,
            "tier_counts": summary.get("tier_counts"),
            "a5_scope_records": a5_records,
            "outside_a5_scope_records": (total - a5_records
                                         if a5_records is not None else None),
            "captures_total": captures_total,
            "capture_count_source": capture_count_source,
            "a5_capture_count": a5_capture_count,
            "legacy_a5_unique_event_captures": legacy_captures,
            "legacy_a5_duplicate_extra_records": legacy_duplicate_extra,
            "legacy_a5_duplicate_groups": legacy_duplicate_groups,
            "a5_duplicate_identity_check": identity_counts.get("identity_check"),
            "source_set_duplicate_groups": _int_field(
                summary, "source_set_duplicate_groups"),
            "source_set_duplicate_extra_records": _int_field(
                summary, "source_set_duplicate_extra_records"),
            "source_relations_total": _int_field(summary, "source_relations_total"),
            "multi_source_records": _int_field(summary, "multi_source_records"),
            "derived_records": _int_field(summary, "derived_records"),
            "a5_native_case_count": native_case_count,
            "a5_native_cohort_records": native_cohort_records,
            "a5_records_with_native_evidence": _int_field(
                summary, "a5_records_with_native_evidence"),
            "a5_records_context_unobserved": _int_field(
                summary, "a5_records_context_unobserved"),
            "a5_records_runtime_state_not_restored": _int_field(
                summary, "a5_records_runtime_state_not_restored"),
            "a5_records_source_binding_ambiguous": _int_field(
                summary, "a5_records_source_binding_ambiguous"),
        },
        "unrecovered": {
            "runtime_state_not_restored_records": _int_field(
                summary, "a5_records_runtime_state_not_restored"),
            "context_unobserved_records": _int_field(
                summary, "a5_records_context_unobserved"),
            "source_binding_ambiguous_records": _int_field(
                summary, "a5_records_source_binding_ambiguous"),
            "historical_absolute_15us_is_not_this_round": True,
        },
        "sources": {"inventory_summary": summary_source},
        "identity_ledger": None,
        "a5_record_recovery": None,
        "capture_recovery": None,
    }
    if capture_report is not None:
        result["capture_recovery"] = {
            "capture_count": captures_total,
            "capture_count_source": capture_count_source,
            "tier_capture_counts": capture_report.get("tier_capture_counts"),
            "disposition_counts": capture_report.get("disposition_counts"),
            "schema_family_counts": capture_report.get("schema_family_counts"),
            "sidecar_candidate_calls_total": capture_report.get(
                "sidecar_candidate_calls_total"),
            "full_scan_verified_calls_total": capture_report.get(
                "full_scan_verified_calls_total"),
            "lower_bound_calls_total": capture_report.get("lower_bound_calls_total"),
            "upper_bound_calls_total": capture_report.get("upper_bound_calls_total"),
            "captures_with_full_scan": capture_report.get("captures_with_full_scan"),
            "captures_with_sidecar": capture_report.get("captures_with_sidecar"),
            "captures_with_abi": capture_report.get("captures_with_abi"),
            "captures_empty": capture_report.get("captures_empty"),
            "native_archive": capture_report.get("native_archive"),
            "recovery_depth_counts": capture_report.get("recovery_depth_counts"),
            "gap_counts": capture_report.get("captures_gap_counts"),
            "native_cohorts_validated": _int_field(
                capture_report, "native_cohorts_validated"),
            "native_windows_reproduced": _int_field(
                capture_report, "native_windows_reproduced"),
            "comparable_calls_lower_bound": capture_report.get(
                "comparable_calls_lower_bound"),
            "comparable_calls_upper_bound": capture_report.get(
                "comparable_calls_upper_bound"),
            "captures_step_values": capture_report.get("captures_step_values"),
        }
        result["sources"]["captures_summary"] = capture_source
    if manifest_report is not None:
        result["capture_manifest"] = {
            "capture_count": captures_total,
            "bucket_count": _int_field(manifest_report, "bucket_count"),
            "total_lines": _int_field(manifest_report, "total_lines"),
            "shard_count": len(manifest_report.get("shard_files") or ()),
        }
        result["sources"]["captures_manifest"] = manifest_source
    if ledger is not None:
        if a5_records is None:
            raise ValueError("inventory_ledger_requires_a5_record_count")
        ledger_report = _ledger_identity(ledger, total)
        if ledger_report["a5_rows"] != a5_records:
            raise ValueError("inventory_ledger_a5_rows_do_not_match_summary")
        if legacy_duplicate_extra is not None \
                and ledger_report["a5_duplicate_extra_records"] != legacy_duplicate_extra:
            raise ValueError("inventory_ledger_a5_duplicate_extra_mismatch")
        if legacy_duplicate_groups is not None \
                and ledger_report["a5_duplicate_groups"] != legacy_duplicate_groups:
            raise ValueError("inventory_ledger_a5_duplicate_groups_mismatch")
        if legacy_captures is not None \
                and ledger_report["a5_captures"]["distinct"] != legacy_captures:
            raise ValueError("inventory_ledger_a5_captures_mismatch")
        result["identity_ledger"] = ledger_report
        result["sources"]["identity_ledger"] = ledger_report["source"]
    if records is not None:
        if a5_records is None:
            raise ValueError("inventory_records_requires_a5_record_count")
        recovery = _a5_record_recovery(records, a5_records)
        result["a5_record_recovery"] = recovery
        result["sources"]["a5_records"] = recovery["source"]
        flags = recovery["flags"]
        result["unrecovered"].update({
            "runtime_state_not_restored_records": flags[
                "per_step_runtime_state_restored"].get("false", 0),
            "context_unobserved_records": flags[
                "context_explicitly_unobserved"].get("true", 0),
            "source_binding_ambiguous_records": flags[
                "source_binding_ambiguous"].get("true", 0),
            "full_per_call_10pct_validated_true_records": flags[
                "full_per_call_10pct_validated"].get("true", 0),
        })
    return result


def comparison_coverage(inventory, acceptance, *, linkage=None):
    """Keep the comparison denominator distinct from the 7337-ledger inventory."""
    if not isinstance(inventory, Mapping) or not isinstance(acceptance, Mapping):
        raise ValueError("comparison_coverage_requires_inventory_and_acceptance")
    ledger = inventory.get("ledger")
    counts = acceptance.get("counts")
    if not isinstance(ledger, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("comparison_coverage_inputs_are_incomplete")
    total = ledger.get("total_records")
    return {
        "schema": COMPARISON_COVERAGE_SCHEMA,
        "ledger_records": total,
        "ledger_a5_records": ledger.get("a5_scope_records"),
        "ledger_captures_total": ledger.get("captures_total"),
        "ledger_a5_event_captures": (ledger.get("a5_unique_event_captures")
                                     if ledger.get("a5_unique_event_captures") is not None
                                     else ledger.get("a5_capture_count")),
        "comparison": {
            "paired_items": counts.get("paired_items"),
            "duration_denominator_items": counts.get("duration_denominator_items"),
            "work_only": counts.get("work_only"),
            "within": counts.get("within"),
            "outside": counts.get("outside"),
            "observation_samples": counts.get("observation_samples"),
            "unpaired_observations": counts.get("unpaired_observations"),
            "unpaired_predictions": counts.get("unpaired_predictions"),
            "ambiguous": counts.get("ambiguous"),
            "condition_gap_items_in_denominator": counts.get(
                "condition_gap_items_in_denominator"),
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
        },
        "source_coverage": {
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
            "window": acceptance.get("window"),
            "window_count": len(acceptance.get("windows") or ()),
            "source_identity": acceptance.get("source_identity"),
            "observation_counts": acceptance.get("observation_counts"),
        },
        "inventory_linkage": (
            {
                "status": linkage["status"],
                "record_match_scope": "ledger_identity_link_only_not_call_certification",
                "paired_items": linkage["comparison"]["paired_items"],
                "matched_items": linkage["comparison"]["matched_items"],
                "unmatched_items": linkage["comparison"]["unmatched_items"],
                "conflict_items": linkage["comparison"]["conflict_items"],
                "matched_inventory_records": linkage["independent_evidence"][
                    "matched_audit_records"],
                "independent_capture_count": linkage["independent_evidence"][
                    "independent_capture_count"],
                "independent_source_set_count": linkage["independent_evidence"][
                    "independent_source_set_count"],
                "duplicate_reference_records": linkage["independent_evidence"][
                    "duplicate_reference_records"],
                "derived_records": linkage["independent_evidence"]["derived_records"],
                "precision_denominator_paired_items": linkage["independent_evidence"][
                    "precision_denominator_paired_items"],
                "duration_denominator_items": linkage["independent_evidence"][
                    "duration_denominator_items"],
                "paired_items_retained": linkage["independent_evidence"][
                    "paired_items_retained"],
                "work_only_items_retained": linkage["independent_evidence"][
                    "work_only_items_retained"],
                "record_match_changes_precision_denominator": False,
            }
            if linkage is not None else {
                "status": "not_declared",
                "reason": "record_dispositions_not_supplied",
                "matched_inventory_records": 0,
            }),
        "validated_fraction_of_ledger": None,
        "ledger_records_not_claimed_by_this_slice": total,
        "statement": (
            "This independent native slice validates only its duration-scored "
            "paired items under the source formal status and coverage. Work-only "
            "ABI pairs are retained as associations but add no duration verdict. "
            "Record matching is identity linkage, not per-call certification; it "
            "is not a 7337-record, duplicate-collapsed or heldout accuracy "
            "fraction."),
    }
