"""Ascend A5 communication expansion and contention calibration.

[Role] Expose strictly keyed A5 UB-mesh/URMA service curves, an expansion
  regime resolver, MC2 chain overlap summaries, one native AI_CPU child local
  service and engine-binding helpers.
[Invariants] A curve is keyed by (collective, world_size, placement_scope,
  dtype, expansion regime, pair, stack, CANN, algorithm, protocol).  A requested
  AICPU/AIV mode is never promoted without observed or profiler evidence.
  Incompatible software stacks are never pooled.  Non-A5 hardware is rejected.
  Single-launch anchors never enter a fit.  A measured phase (AI_CPU child,
  resource phase, complete collective call, directed p2p) stays distinct from
  the consumer role and is never a per-hop link price.  A supply change must
  match the measured SKU/variant/topology/stack/CANN/direction/shared-domain/
  transport window and the known software versions; source lineage is compared
  separately from target applicability.
[Timeouts & Locks] No timers, locks or subprocesses.  Ingestion of an explicit
  path is a bounded synchronous or ijson-streamed JSON read.
[Side-Effect Policy] Pure computation.  Never launches a device, writes
  calibration or mutates the primitive store.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import math
from statistics import median
from typing import Mapping

from mindie_sim.prediction.services import ServiceCurve
from mindie_sim.schema.common import ModelingError
from .a5_comm_binding import (
    _SUPPORTED_COLLECTIVES, _fold_timings, _weighted_median, bind_service_curve,
    build_a5_comm_recipe, to_service_regime,
)
from .a5_comm_golden import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_NATIVE_GROUP2, A5_SKU,
    EMBEDDED_A5_CAMPAIGNS, EMBEDDED_A5_MC2_CAMPAIGNS,
)
from .a5_comm_ingest import (
    _coerce_ingested, _collective, _dtype, _explicit_pair, _explicit_scope,
    _fallback_measurement, _fallback_transport_window,
    _is_a5, _mode, _protocol_key, _require_a5,
    _validate_measurement_window,
    ingest_a5_campaign_data as _ingest_a5_campaign_data,
    resolve_expansion_regime as _resolve_expansion_regime,
)
from .resource_services import normalize_resource_service
from .service_fitting import fit_resource_service, fit_service_curve

__all__ = [
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU", "A5CommCalibrationBundle",
    "A5_COMM_LOCAL_SERVICE_SCHEMA", "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "a5_local_service_applicability_key", "a5_local_service_domain_key",
    "a5_local_service_lineage_key", "a5_local_service_prediction",
    "a5_local_service_supply_change", "bind_service_curve", "build_a5_comm_recipe",
    "build_a5_followup_manifest", "collect_native_service",
    "compute_a5_contention_slowdown", "evaluate_a5_mc2_chains",
    "fit_a5_native_local_service", "fit_a5_service_curves",
    "ingest_a5_campaign_data", "native_child_stage_evidence",
    "require_a5_local_service_applicability",
    "require_a5_local_service_domain", "resolve_expansion_regime",
    "to_service_regime",
]

A5_COMM_LOCAL_SERVICE_SCHEMA = "a5-comm-local-service/2"
A5_NATIVE_CHILD_EVIDENCE_SCHEMA = "a5-native-child-stage-evidence/1"
_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_PAIR_ORDER = {"0->1": 0, "0->4": 1, "0->8": 2, "0->24": 3}
# Source direction of the data movement each collective performs.  AllGather
# channels read peers; AllReduce/ReduceScatter steps and AllToAll exchanges
# write/send.  This is the HCCL contract direction, not a measured link sample.
_COLLECTIVE_DIRECTION = {
    "all_gather": "read", "all_reduce": "send", "reduce_scatter": "send",
    "all_to_all": "send", "all_to_allv": "send", "p2p_send_recv": "directed",
}
_NATIVE_LOCAL_DIRECTION = {"all_gather": "read", "reduce_scatter": "send"}
_NATIVE_LOCAL_COLLECTIVES = tuple(_NATIVE_LOCAL_DIRECTION)
_NATIVE_LOCAL_COLLECTIVE_SET = frozenset(_NATIVE_LOCAL_COLLECTIVES)
_PRIMARY_STACKS = ("rack06", "rack06-07", "rack06-09")
_AMBIGUOUS = "a5_incompatible_software_stacks_require_selector"
def resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode, *,
                             profiler_evidence=None):
    """Resolve one expansion regime without promoting an unconfirmed request."""
    return _resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode,
                                     profiler_evidence=profiler_evidence)
def ingest_a5_campaign_data(source=None):
    """Normalize A5 campaigns into steady-state conditions and launch anchors."""
    return _ingest_a5_campaign_data(source)


def _data_for(ingested):
    return ingest_a5_campaign_data() if ingested is None else _coerce_ingested(ingested)


def _software_key(row):
    return (row.get("stack"), row.get("cann"), row.get("algorithm"),
            _protocol_key(row.get("protocol")))


def _sample_split(row, holdout_payloads):
    split = row.get("split")
    if split is None:
        return "heldout" if row["payload_bytes"] in holdout_payloads else "train"
    if split not in {"train", "heldout"}:
        raise ValueError("service_sample_split_requires_train_or_heldout")
    return split


def _curve_identity(collective, world_size, scope, dtype, regime, pair, evidence,
                    software):
    return {
        "hardware": {"key": A5_HARDWARE_KEY, "sku": A5_SKU,
                     "variant": "DT",
                     "architecture": A5_ARCH_FAMILY},
        "transport": "ub_mesh" if scope.startswith("intra") else "urma",
        "collective": collective, "world_size": int(world_size),
        "placement_scope": scope, "dtype": dtype,
        "expansion_regime": regime, "p2p_pair": pair, "evidence": evidence,
        "stack": software[0], "cann": software[1], "algorithm": software[2],
        "protocol": software[3],
    }


def _shared_domain(scope, world_size, pair=None):
    """One physical sharing domain, not a per-call duration or hop price.

    A collective's members share the same intra-node UB mesh or the same
    cross-node URMA fabric.  A direct pair names its exact directed pair.
    """
    intra = str(scope).startswith("intra")
    domain = {
        "id": (f"{'ub_mesh' if intra else 'urma'}:{scope}:W{int(world_size)}"
               + (f":{pair}" if pair else "")),
        "kind": "ub_mesh" if intra else "urma",
        "scope": str(scope),
        "members": int(world_size),
        "inter_node": not intra,
    }
    if pair:
        domain["pair"] = str(pair)
    return domain


def _measurement_from_rows(rows, collective, boundary):
    """Canonicalize every row's measured phase before comparing/aggregating.

    Native extraction attaches ``measurement`` to every condition.  Legacy
    campaign rows without one get a labeled campaign fallback **per row**, so
    a mixed explicit/fallback group cannot silently inherit the explicit row's
    phase.  No row is classified as native by a path suffix or campaign name.
    """
    canonical = []
    for row in rows:
        declared = row.get("measurement")
        if isinstance(declared, Mapping):
            measurement = dict(declared)
            measurement.setdefault("explicit", True)
        else:
            measurement = _fallback_measurement(
                collective, boundary, row.get("algorithm") is not None)
        canonical.append(measurement)
    first = canonical[0]
    if any(item != first for item in canonical):
        raise ValueError("a5_measurement_boundary_mixed_sources")
    return first


def _transport_window(rows, collective, boundary):
    """Canonicalize every row's window; a missing row gets its own fallback."""
    canonical = []
    for row in rows:
        declared = row.get("transport_window")
        if isinstance(declared, Mapping):
            window = dict(declared)
        else:
            measurement = row.get("measurement")
            if not isinstance(measurement, Mapping):
                measurement = _fallback_measurement(
                    collective, boundary, row.get("algorithm") is not None)
            window = _fallback_transport_window(dict(measurement))
        canonical.append(window)
    first = canonical[0]
    if any(item != first for item in canonical):
        raise ValueError("a5_transport_window_mixed_sources")
    measurement = rows[0].get("measurement")
    if not isinstance(measurement, Mapping):
        measurement = _fallback_measurement(
            collective, boundary, rows[0].get("algorithm") is not None)
    _validate_measurement_window(dict(measurement), first)
    return first


def _measurement_boundary(collective, boundary, native=False,
                         algorithm_identified=False):
    """Describe the measured interval, not the consumer's later role.

    A native ``HcclLaunchAicpuKernel`` AI_CPU child is not a complete
    collective call.  A campaign per-op collective is a complete measured call
    envelope.  A directed p2p call is a complete directed call.  An explicit
    resource-phase row is a phase, not a device call.  Binding and export must
    preserve this descriptor instead of relabelling every collective.
    """
    if native:
        return {
            "kind": "native_target_call", "phase": "aicpu_child",
            "resource": "aicpu_child",
            "work_coordinate": "per_rank_input_bytes",
            "duration_unit": "ms", "source_kind": "native_ai_cpu_child",
            "service_semantics": "aicpu_child_collective_phase",
            "complete_collective_call": False,
            "complete_directed_call": False,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False,
        }
    if boundary == "resource_phase":
        return {
            "kind": "resource_phase", "phase": "collective_phase",
            "resource": "collective_phase",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_resource_phase",
            "service_semantics": "collective_phase_service",
            "complete_collective_call": False,
            "complete_directed_call": False,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False,
        }
    if collective == "p2p_send_recv":
        return {
            "kind": "device_call", "phase": "directed_p2p_call",
            "resource": "directed_p2p_call",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_device_call",
            "service_semantics": "directed_p2p_call",
            "complete_collective_call": False,
            "complete_directed_call": True,
            "link_transit_identified": False,
            "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": True,
        }
    return {
        "kind": "device_call", "phase": "collective_call",
        "resource": "collective_call",
        "work_coordinate": "captured_per_rank_payload",
        "duration_unit": "ms", "source_kind": "campaign_device_call",
        "service_semantics": "whole_collective_call_envelope",
        "complete_collective_call": True,
        "complete_directed_call": False,
        "link_transit_identified": False,
        "hop_scaling": "not_derived",
        "algorithm_identified": bool(algorithm_identified),
        "direction_observed": False,
    }


def _service_direction(collective, pair):
    if collective == "p2p_send_recv":
        return str(pair) if pair else "directed"
    return _COLLECTIVE_DIRECTION.get(collective, "unspecified")


def _fit_group(rows, holdout_payloads, key, evidence, extra_identity=None):
    collective, world_size, scope, dtype, regime, pair = key[:6]
    software = key[6:10] if len(key) > 6 else key[6:]
    boundary = (key[10] if len(key) > 10 else
                "resource_phase" if any(row.get("boundary") == "resource_phase"
                                        for row in rows) else "device_kernel")
    by_run, samples = defaultdict(list), []
    for row in rows:
        by_run[(row["payload_bytes"], row.get("run_id") or row["campaign"],
                _sample_split(row, holdout_payloads))].append(row)
    for (work, run_id, split), grp in by_run.items():
        if any("_weights" in r for r in grp):
            vals = []
            for r in grp:
                for val, wt in (r.get("_weights") or ((r["per_op_ms"], 1),)):
                    _fold_timings(vals, val, wt)
            r0 = min(grp, key=lambda r: int(r.get("repeat_index") or 0))
            samples.append({"work": work, "duration_ms": _weighted_median(vals),
                            "run_id": run_id, "repeat_index": int(r0.get("repeat_index") or 0),
                            "source": r0.get("source") or f"{r0['campaign']}:{collective}:W{world_size}:{work}",
                            "split": split})
        else:
            samples.extend({"work": work, "duration_ms": r["per_op_ms"], "run_id": run_id,
                            "repeat_index": int(r.get("repeat_index") or 0),
                            "source": r.get("source") or f"{r['campaign']}:{collective}:W{world_size}:{work}",
                            "split": split} for r in grp)
    train = {sample["work"] for sample in samples if sample["split"] == "train"}
    if len(train) < 2:
        return None, None, "a5_curve_requires_two_training_payloads"
    identity = _curve_identity(collective, world_size, scope, dtype, regime, pair,
                               evidence, software)
    identity["protocol"] = next((row.get("protocol") for row in rows
                                 if row.get("protocol") is not None), software[3])
    direction = _service_direction(collective, pair)
    shared = _shared_domain(scope, world_size, pair)
    measurement = _measurement_from_rows(rows, collective, boundary)
    window = _transport_window(rows, collective, boundary)
    semantics = measurement["service_semantics"]
    algorithm_identified = bool(measurement.get(
        "algorithm_identified", any(row.get("algorithm") is not None
                                    for row in rows)))
    identity.update({"direction": direction, "shared_domain": shared,
                     "transport_window": window,
                     "measurement_boundary": measurement,
                     "service_semantics": semantics,
                     "work_scope": "captured_per_rank_payload"})
    if extra_identity:
        identity.update(extra_identity)
    provenance = {
        "hardware_key": A5_HARDWARE_KEY, "sku": A5_SKU, "arch_family": A5_ARCH_FAMILY,
        "hardware_variant": "DT",
        "campaigns": sorted({row["campaign"] for row in rows}),
        "evidence": evidence, "holdout_payloads": sorted(holdout_payloads),
        "expansion_regime": regime,
        "expansion_unconfirmed": any(row["expansion_unconfirmed"] for row in rows),
        "p2p_pair": pair, "stack": software[0], "cann": software[1],
        "algorithm": software[2],
        "protocol": next((row.get("protocol") for row in rows if row.get("protocol") is not None), None),
        "topology": {"placement_scope": scope, "world_size": int(world_size),
                     "transport": identity["transport"]},
        "software": {"stack": software[0], "cann": software[1],
                     "algorithm": software[2], "protocol": software[3]},
        "direction": direction, "shared_domain": shared,
        "transport_window": window, "service_semantics": semantics,
        "measurement_boundary": measurement,
        "work_scope": "captured_per_rank_payload",
        # A measured call/phase is a completion envelope, not a per-hop link
        # price.  Whether a link transit rate is identifiable is recorded
        # explicitly instead of deriving one by multiplying hops.
        "link_transit_identified": measurement["link_transit_identified"],
        "hop_scaling": measurement["hop_scaling"],
        "direction_basis": ("declared_directed_pair" if collective == "p2p_send_recv"
                            else "hccl_source_template_native_direction"),
        "direction_observed": measurement["direction_observed"],
        "algorithm_identified": measurement["algorithm_identified"],
        "shared_domain_basis": "declared_placement_scope_and_group",
        "transport_window_basis": "source_artifact_kind",
    }
    if extra_identity and extra_identity.get("derived_from"):
        provenance["derived_from"] = extra_identity["derived_from"]
    provenance["boundary"] = boundary
    packet = {"resource": measurement["resource"],
              "work_unit": "bytes", "identity": identity,
              "boundary": boundary, "active_units": max(1, int(world_size)),
              "samples": samples, "provenance": provenance}
    fitted = fit_service_curve(packet)
    return ServiceCurve.from_mapping(fitted["model"]), fitted, None


def _selector_match(curve, selectors):
    for name, wanted in selectors.items():
        if wanted is None:
            continue
        if name == "protocol":
            if _protocol_key(curve.identity.get(name)) != _protocol_key(wanted):
                return False
        elif name == "pair":
            if curve.identity.get("p2p_pair") != wanted:
                return False
        elif name == "boundary":
            if curve.boundary != wanted:
                return False
        elif curve.identity.get(name) != wanted:
            return False
    return True


def _select_software(cell, options):
    """Pick the primary 9.1.0 rack06 stack, or reject an ambiguous pool."""
    primary = [option for option in options if option[0][0] in _PRIMARY_STACKS
               and option[0][1] in {None, "9.1.0"}]
    if primary:
        primary.sort(key=lambda option: (option[0][1] != "9.1.0",
                                         _PRIMARY_STACKS.index(option[0][0]), str(option[0])))
        return primary[0][1]
    if len(options) == 1:
        return options[0][1]
    raise ModelingError(_AMBIGUOUS)


@dataclass(frozen=True)
class A5CommCalibrationBundle:
    """Strictly keyed A5 service curves plus their independent fit evidence."""

    curves: Mapping
    fits: tuple = ()
    holdout: tuple = ()
    single_op_anchors: tuple = ()
    campaigns: tuple = ()
    unfitted: tuple = ()
    provenance: Mapping = field(default_factory=dict)
    hardware_key: str = A5_HARDWARE_KEY
    arch_family: str = A5_ARCH_FAMILY
    stack_curves: Mapping = field(default_factory=dict)

    @property
    def cells(self):
        return tuple(sorted(self.curves))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) >= 6 and key in self.stack_curves:
                return self.stack_curves[key]
            if len(key) == 5:
                return self.curves[key]
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _candidates(self, collective, world_size, *, placement_scope=None,
                    dtype="bf16", expansion_mode="unspecified",
                    allow_unspecified_fallback=False):
        name = _collective(collective)
        if placement_scope is None:
            scope = ("cross_node_urma" if int(world_size) > 8
                     else "intra_node_ub")
        else:
            scope = _explicit_scope(placement_scope)
            if scope is None:
                raise ModelingError(
                    f"a5_placement_scope_unknown:{placement_scope}")
        base = (name, int(world_size), scope, _dtype(dtype))
        mode = _mode(expansion_mode)
        if mode in _EXPANSION_MODES:
            keys = [base + (mode,)]
            if allow_unspecified_fallback:
                keys.append(base + ("unspecified",))
            return keys, mode
        if mode != "unspecified":
            raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
        return [base + ("unspecified",)], mode

    def _pick(self, candidates, selectors, mode):
        requested = any(value is not None for value in selectors.values())
        for candidate in candidates:
            direct = {curve.model_id: curve for key, curve in self.curves.items()
                      if key == candidate and _selector_match(curve, selectors)}
            if not requested and direct:
                return next(iter(direct.values()))
            matches = {curve.model_id: curve for key, curve
                       in self.stack_curves.items() if key[:5] == candidate
                       and _selector_match(curve, selectors)}
            matches.update(direct)
            if not matches:
                continue
            if len(matches) > 1 and len(direct) == 1:
                return next(iter(direct.values()))
            if len(matches) > 1 and selectors.get("boundary") is None:
                device = {curve.model_id: curve for curve in matches.values()
                          if curve.boundary == "device_kernel"}
                if len(device) == 1:
                    return next(iter(device.values()))
            if len(matches) > 1:
                raise ModelingError(_AMBIGUOUS)
            return next(iter(matches.values()))
        if mode in _EXPANSION_MODES:
            raise ModelingError(f"a5_expansion_regime_unconfirmed:{mode}")
        base = candidates[0]
        raise ModelingError(
            f"a5_curve_unavailable:{base[0]}:W{base[1]}:{base[2]}:{base[3]}")

    def lookup_curve(self, collective, world_size=None, *,
                     placement_scope=None, dtype="bf16",
                     expansion_mode="unspecified", hardware_key=A5_HARDWARE_KEY,
                     arch_family=None, allow_unspecified_fallback=False,
                     stack=None, cann=None, algorithm=None, protocol=None,
                     pair=None, boundary=None):
        _require_a5(hardware_key, arch_family)
        selectors = {"stack": stack, "cann": cann, "algorithm": algorithm,
                     "protocol": protocol}
        if boundary is not None:
            selectors["boundary"] = boundary
        if pair is not None:
            pair = _explicit_pair(pair) or str(pair).strip()
            selectors["pair"] = pair
        if isinstance(collective, (tuple, list)):
            key = tuple(collective)
            if len(key) == 6:
                selectors["stack"] = selectors["stack"] or key[5]
                key = key[:5]
            if len(key) != 5:
                raise ModelingError("a5_curve_key_requires_five_or_six_fields")
            if key in self.curves and not any(value is not None
                                              for value in selectors.values()):
                return self.curves[key]
            candidates, mode = self._candidates(
                key[0], key[1], placement_scope=key[2], dtype=key[3],
                expansion_mode=key[4], allow_unspecified_fallback=allow_unspecified_fallback)
            return self._pick(candidates, selectors, mode)
        candidates, mode = self._candidates(
            collective, world_size, placement_scope=placement_scope, dtype=dtype,
            expansion_mode=expansion_mode,
            allow_unspecified_fallback=allow_unspecified_fallback)
        if pair and candidates:
            expanded = []
            inferred = {"0->24": "cross_node_urma_far",
                        "0->1": "intra_node_ub"}.get(pair)
            for base in candidates:
                suffix = f"{base[2]}_{pair.replace('->', '_')}"
                if inferred and (placement_scope is None or (
                        pair == "0->24" and base[2] == "cross_node_urma")):
                    expanded.append(base[:2] + (inferred,) + base[3:])
                expanded.append(base[:2] + (suffix,) + base[3:])
                expanded.append(base)
            candidates = expanded
        return self._pick(candidates, selectors, mode)

    def to_dict(self):
        return {
            "schema": "a5-comm-calibration-bundle/1",
            "hardware_key": self.hardware_key, "arch_family": self.arch_family,
            "curves": {":".join((key[0], f"W{key[1]}", key[2], key[3], key[4])): curve.to_dict()
                       for key, curve in sorted(self.curves.items(), key=lambda item: str(item[0]))},
            "stack_curves": {":".join(str(part) for part in
                                      (key[0], f"W{key[1]}", key[2], key[3], key[4], *key[5:])):
                             curve.to_dict() for key, curve in
                             sorted(self.stack_curves.items(), key=lambda item: str(item[0]))},
            "fits": list(self.fits), "holdout": list(self.holdout),
            "single_op_anchors": list(self.single_op_anchors),
            "campaigns": list(self.campaigns), "unfitted": list(self.unfitted),
            "provenance": dict(self.provenance)}

    def bind_service_curve(self, case, op, scope="none", *,
                           require_complete_call=False, production=False,
                           **lookup_kwargs):
        return bind_service_curve(
            self, case, op, scope,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)

    def to_service_regime(self, case, op, scope="none", *,
                          require_complete_call=False, production=False,
                          **lookup_kwargs):
        return to_service_regime(
            self, case, op, scope,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)

    def build_a5_comm_recipe(self, collective=None, *,
                             allow_unspecified_fallback=False,
                             require_complete_call=False, production=False,
                             **lookup_kwargs):
        return build_a5_comm_recipe(
            self, collective,
            allow_unspecified_fallback=allow_unspecified_fallback,
            require_complete_call=require_complete_call,
            production=production, **lookup_kwargs)


def fit_a5_service_curves(ingested=None, *, holdout_payloads=(65536, 16777216)):
    """Fit one ServiceCurve per strict A5 communication software partition."""
    data = _data_for(ingested)
    _require_a5(data.get("hardware_key") or data.get("sku") or A5_HARDWARE_KEY, data.get("arch_family"))
    holdout = frozenset(int(payload) for payload in holdout_payloads)
    groups = defaultdict(list)
    for row in data.get("conditions") or ():
        if row["single"] or row["collective"] not in _SUPPORTED_COLLECTIVES:
            continue
        key = (row["collective"], int(row["world_size"]), row["placement_scope"],
               row["dtype"], row["regime"],
               row["pair"] if row["collective"] == "p2p_send_recv" else None,
               *_software_key(row), row.get("boundary") or "device_kernel")
        groups[key].append(row)
    derived_keys = set()
    for key, rows in list(groups.items()):
        if key[0] != "all_to_all":
            continue
        target = ("all_to_allv",) + key[1:2] + key[2:]
        if target not in groups:
            groups[target] = rows
            derived_keys.add(target)
    curves, fits, holdout_rows, unfitted = {}, [], [], []
    software_maps = defaultdict(lambda: defaultdict(list))
    for key in sorted(groups, key=str):
        rows = groups[key]
        derived = key in derived_keys
        evidence = "derived_all_to_all_equivalent" if derived else "measured"
        extra = {"derived_from": "all_to_all"} if derived else None
        curve, fitted, error = _fit_group(rows, holdout, key, evidence, extra)
        if error:
            unfitted.append({"key": key, "reason": error})
            continue
        cell = key[:5]
        software_maps[key[6:10]][cell].append((key[5], curve, fitted, key))
        fits.append({**fitted, "cell": cell, **_cell_fields(cell),
                     "p2p_pair": key[5], "evidence": evidence,
                     "stack": key[6], "cann": key[7], "algorithm": key[8],
                     "boundary": key[10]})
        holdout_rows.extend({**row, "cell": cell, **_cell_fields(cell),
                             "p2p_pair": key[5], "split": "heldout"}
                            for row in fitted["heldout"])
    stack_curves = {}
    target_options = defaultdict(lambda: defaultdict(list))
    for software in sorted(software_maps, key=str):
        for cell, variants in software_maps[software].items():
            by_boundary = defaultdict(list)
            for entry in variants:
                by_boundary[entry[3][10]].append(entry)
            for boundary in sorted(by_boundary):
                boundary_variants = by_boundary[boundary]
                boundary_variants.sort(key=lambda value: (
                    _PAIR_ORDER.get(value[0], 99), str(value[3])))
                targets = defaultdict(list)
                for index, entry in enumerate(boundary_variants):
                    pair = entry[0]
                    if index == 0:
                        targets[cell].append(entry)
                        continue
                    suffix = pair.replace("->", "_")
                    targets[(cell[0], cell[1], f"{cell[2]}_{suffix}", cell[3],
                             cell[4])].append(entry)
                    if pair == "0->24":
                        targets[(cell[0], 24, cell[2], cell[3], cell[4])].append(entry)
                        targets[(cell[0], cell[1], "cross_node_urma_far", cell[3],
                                 cell[4])].append(entry)
                for target, entries in targets.items():
                    curve = entries[0][1]
                    stack_curves.setdefault(target + (software[0],), curve)
                    if boundary == "resource_phase":
                        stack_curves.setdefault(
                            target + (software[0], boundary), curve)
                    full_key = target + software
                    if boundary == "device_kernel" or full_key not in stack_curves:
                        stack_curves[full_key] = curve
                    else:
                        stack_curves[full_key + (boundary,)] = curve
                    target_options[target][boundary].append((software, curve))
    for cell, options_by_boundary in target_options.items():
        options = (options_by_boundary.get("device_kernel")
                   or options_by_boundary.get("resource_phase") or ())
        if options:
            curves[cell] = _select_software(cell, options)
    provenance = {
        "hardware_key": A5_HARDWARE_KEY, "sku": A5_SKU,
        "arch_family": A5_ARCH_FAMILY,
        "source_campaigns": list(data.get("campaign_labels") or ()),
        "holdout_payloads": sorted(holdout),
        "curve_count": len(curves), "stack_curve_count": len(stack_curves),
        "fit_count": len(fits), "unfitted_count": len(unfitted),
    }
    return A5CommCalibrationBundle(
        curves=curves, fits=tuple(fits), holdout=tuple(holdout_rows),
        single_op_anchors=tuple(data.get("single_op_anchors") or ()),
        campaigns=tuple(data.get("campaigns") or ()),
        unfitted=tuple(unfitted), provenance=provenance,
        stack_curves=stack_curves)


def _cell_fields(key):
    return dict(zip(("collective", "world_size", "placement_scope", "dtype",
                     "expansion_mode"), key[:5]))


def compute_a5_contention_slowdown(expansion_mode, *, concurrent_cube_util=0.0,
                                   concurrent_vector_util=0.0,
                                   concurrent_mte_util=0.0, active_aiv_blocks=16,
                                   total_aiv_units=64):
    """Shared-vector / MTE / HBM contention for an A5 execution mode."""
    mode = _mode(expansion_mode)
    if mode not in _EXPANSION_MODES and mode != "unspecified":
        raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
    for name, value in (("concurrent_cube_util", concurrent_cube_util),
                        ("concurrent_vector_util", concurrent_vector_util),
                        ("concurrent_mte_util", concurrent_mte_util)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"a5_contention_util_invalid:{name}")
    if int(total_aiv_units) < 1:
        raise ValueError("a5_contention_requires_positive_total_aiv_units")
    blocks = max(0, int(active_aiv_blocks))
    units = max(1, int(total_aiv_units))
    occupancy = min(1.0, blocks / units)
    cube = min(4.0, float(concurrent_cube_util))
    vector_demand = min(4.0, float(concurrent_vector_util))
    mte = min(4.0, float(concurrent_mte_util))
    mte_slowdown = 1.0 + 0.55 * mte + 0.10 * vector_demand + 0.05 * cube
    hbm_slowdown = 1.0 + 0.45 * mte + 0.15 * vector_demand + 0.20 * cube
    cube_slowdown = 1.0 + 0.35 * cube + 0.05 * mte
    vector_slowdown = 1.0
    if mode == "aiv":
        # AIV owns the vector pipe; more resident blocks and more concurrent
        # vector demand extend the same ALU service time.
        vector_slowdown = (1.0 + vector_demand * (0.55 + 0.45 * occupancy)
                           + 0.10 * cube)
    total = max(vector_slowdown, mte_slowdown, hbm_slowdown, cube_slowdown)
    transfer = 0.012 + 0.05 * mte + 0.01 * vector_demand
    sync = 0.008 + 0.02 * max(mte, vector_demand, cube) + 0.0005 * blocks
    return {
        "expansion_mode": mode,
        "vector_slowdown": vector_slowdown,
        "mte_slowdown": mte_slowdown,
        "hbm_slowdown": hbm_slowdown,
        "cube_slowdown": cube_slowdown,
        "contention_slowdown": total,
        "total_slowdown": total,
        "overall_slowdown": total,
        "vector_alu_slowdown": vector_slowdown,
        "transfer_latency_ms": transfer, "sync_latency_ms": sync,
        "transfer_ms": transfer, "sync_ms": sync,
        "active_aiv_blocks": blocks,
        "total_aiv_units": units,
        "aiv_occupancy": occupancy,
    }


def evaluate_a5_mc2_chains(ingested=None):
    """Summarize decomposed MC2 chain overlap and resolved expansion regime."""
    data = _data_for(ingested)
    rows = []
    for chain in data.get("mc2_chains") or ():
        dtype = _dtype(chain["dtype"])
        legs = chain.get("legs_ms") or {}
        chain_ms = float(chain["chain_p50_ms"])
        sum_ms = float(chain.get("sum_leg_p50_ms") or sum(legs.values()))
        ratio = chain_ms / sum_ms
        concurrent = max(0.01, min(1.0, 1.0 - min(1.0, ratio)))
        if any(key in chain for key in ("observed_expansion_mode",
                                        "requested_expansion_mode",
                                        "profiler_evidence")):
            mode = resolve_expansion_regime(
                chain.get("requested_expansion_mode"),
                chain.get("observed_expansion_mode"),
                profiler_evidence=chain.get("profiler_evidence"))["regime"]
        else:
            mode = _mode(chain.get("regime"))
        contention = compute_a5_contention_slowdown(
            mode, concurrent_cube_util=0.40 * concurrent,
            concurrent_vector_util=0.75 * concurrent,
            concurrent_mte_util=concurrent)
        rows.append({
            **chain, "dtype": dtype, "chain_p50_ms": chain_ms,
            "sum_leg_p50_ms": sum_ms, "overlap_ratio": ratio,
            "chain_over_sum": ratio,
            "expansion_regime": mode,
            "contention_degradation_multiplier": contention["contention_slowdown"],
            "vector_slowdown": contention["vector_slowdown"],
            "mte_slowdown": contention["mte_slowdown"],
            "hbm_slowdown": contention["hbm_slowdown"],
            "contended_chain_ms": chain_ms * contention["contention_slowdown"],
        })
    by_dtype = {}
    for dtype in sorted({row["dtype"] for row in rows}):
        subset = [row for row in rows if row["dtype"] == dtype]
        ratios = sorted(row["overlap_ratio"] for row in subset)
        multipliers = sorted(row["contention_degradation_multiplier"] for row in subset)
        by_dtype[dtype] = {
            "count": len(subset),
            "world_sizes": sorted({row["world_size"] for row in subset}),
            "campaigns": sorted({row["campaign"] for row in subset}),
            "overlap_ratio": {"min": ratios[0], "max": ratios[-1],
                              "median": median(ratios),
                              "mean": sum(ratios) / len(ratios)},
            "contention_degradation_multiplier": {
                "min": multipliers[0], "max": multipliers[-1],
                "median": median(multipliers)},
        }
    quantized = [row for row in rows
                 if row["dtype"] in {"int8", "fp8-e4m3"}]
    ratios = sorted(row["overlap_ratio"] for row in quantized)
    all_ratios = sorted(row["overlap_ratio"] for row in rows)
    multipliers = sorted(row["contention_degradation_multiplier"] for row in quantized)
    summary = {
        "schema": "a5-mc2-chain-evaluation/1", "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "chains": rows, "rows": rows,
        "dtypes": sorted(by_dtype), "by_dtype": by_dtype,
        "overlap_ratio": ({"min": ratios[0], "max": ratios[-1],
                           "median": median(ratios),
                           "mean": sum(ratios) / len(ratios)} if ratios else {}),
        "all_overlap_ratio": (
            {"min": all_ratios[0], "max": all_ratios[-1],
             "median": median(all_ratios),
             "mean": sum(all_ratios) / len(all_ratios)} if all_ratios else {}),
        "contention_degradation": (
            {"min": multipliers[0], "max": multipliers[-1],
             "median": median(multipliers)} if multipliers else {}),
    }
    return summary


def build_a5_followup_manifest(ingested=None):
    """Return a deterministic verified/pending cell manifest."""
    data = _data_for(ingested)
    verified = {}
    for row in data.get("conditions") or ():
        if row["single"]:
            continue
        key = (row["collective"], int(row["world_size"]), row["placement_scope"],
               row["dtype"], row["regime"], row.get("pair"))
        entry = verified.setdefault(key, {"campaigns": set(), "payloads": set(),
                                          "unconfirmed": False})
        entry["campaigns"].add(row["campaign"])
        entry["payloads"].add(row["payload_bytes"])
        entry["unconfirmed"] = entry["unconfirmed"] or row["expansion_unconfirmed"]
    verified_cells = [_ManifestCell({
        "collective": key[0], "world_size": key[1], "placement_scope": key[2],
        "dtype": key[3], "expansion_mode": key[4], "p2p_pair": key[5],
        "campaigns": sorted(value["campaigns"]),
        "payload_count": len(value["payloads"]),
        "expansion_unconfirmed": value["unconfirmed"], "evidence": "measured",
    }) for key, value in sorted(verified.items(), key=lambda item: str(item[0]))]
    has_confirmed = {"aicpu", "aiv"} <= {
        cell["expansion_mode"] for cell in verified_cells}
    has_four_hop = any(
        cell["collective"] == "p2p_send_recv"
        and cell["placement_scope"] == "intra_node_ub"
        and cell.get("p2p_pair") == "0->4" for cell in verified_cells)
    has_w64 = any(cell["world_size"] >= 64 for cell in verified_cells)
    wide_all_to_allv = {cell["world_size"] for cell in verified_cells
                        if cell["collective"] == "all_to_allv"}
    has_wide_all_to_allv = {16, 32} <= wide_all_to_allv
    pending_cells = []
    if not has_confirmed:
        pending_cells.append(_ManifestCell({
            "cell": "expansion_regime:aicpu/aiv",
            "requires": "msprof task-trace confirmation",
            "reason": "no measured row observed an AICPU or AIV expansion engine"}))
    if not has_w64:
        pending_cells.append(_ManifestCell({
            "cell": "collective:W64-8node",
            "requires": "W64 8-node expansion",
            "reason": "W32 is the widest measured mesh; W64 extrapolation is unverified"}))
    if not has_four_hop:
        pending_cells.append(_ManifestCell({
            "cell": "p2p_send_recv:0->4:intra_node_ub",
            "requires": "0->4 intra-node P2P pair",
            "reason": "only the 0->1 intra-node pair is measured"}))
    pending_cells.append(_ManifestCell({
        "cell": "contention:live_sweeps",
        "requires": "live concurrent contention sweeps",
        "reason": "contention multipliers are analytic, not a measured sweep"}))
    if not has_wide_all_to_allv:
        pending_cells.append(_ManifestCell({
            "cell": "all_to_allv:W16-W32",
            "requires": "measured all_to_allv lane",
            "reason": "all_to_allv curves outside W2-W8 are derived from all_to_all"}))
    pending_cells.sort(key=lambda item: item["cell"])
    return {
        "schema": "a5-followup-manifest/1",
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "source_campaigns": sorted(data.get("campaign_labels") or ()),
        "verified_cells": verified_cells,
        "pending_cells": pending_cells,
        "verified_cell_ids": [f"{cell['collective']}:W{cell['world_size']}:"
                              f"{cell['placement_scope']}:{cell['dtype']}:"
                              f"{cell['expansion_mode']}" for cell in verified_cells],
        "pending_cell_ids": [cell["cell"] for cell in pending_cells],
        "summary": {
            "verified_cells": len(verified_cells),
            "pending_cells": len(pending_cells),
            "verified_world_sizes": sorted({cell["world_size"]
                                            for cell in verified_cells}),
            "unconfirmed_conditions": sum(1 for row in data.get("conditions") or ()
                                          if row["single"] is False
                                          and row["expansion_unconfirmed"]),
        },
    }


class _ManifestCell(dict):
    """Structured cell whose membership also matches its readable text."""

    def _text(self):
        return json.dumps(self, sort_keys=True)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        return str(item).lower() in self._text().lower()

    def __eq__(self, other):
        if isinstance(other, str):
            return other in {self.get("cell"), self._text()}
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def lower(self): return self._text().lower()
    def upper(self): return self._text().upper()
    def startswith(self, prefix): return self._text().startswith(prefix)
    def endswith(self, suffix): return self._text().endswith(suffix)


NATIVE_A5_SERVICE_SOURCE = A5_NATIVE_GROUP2["source"]


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_source_identity(path, *, stack, cann, execution_core, placement,
                            deployment_image_id):
    """Validate the native wave's actual source/environment and return it.

    The identity is the environment that produced the observations: manifest
    revision, software packages, host/image and intrahost UB topology.  A
    caller may not respecify any of these and then reuse the fitted supply.
    """
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    source_id = str(summary.get("manifest_sha256") or "")
    manifest = path.parents[2] / "manifest" / "native-manifest.json"
    if not source_id or not manifest.is_file() or _file_sha256(manifest) != source_id:
        raise ValueError("native_service_manifest_identity_mismatch")
    versions = path.parents[1] / "native-versions.json"
    if not versions.is_file():
        raise ValueError("native_service_versions_missing")
    version = json.loads(versions.read_text(encoding="utf-8"))
    packages = version.get("python_packages") or {}
    if str(cann) not in str(packages.get("cann") or "") and str(cann) not in str(
            version.get("cann_version_info") or ""):
        raise ValueError("native_service_cann_mismatch")
    if "intrahost" not in str(version.get("topology_interpretation") or "").lower():
        raise ValueError("native_service_topology_not_intrahost")
    environment = path.parents[1] / "environment.txt"
    if not environment.is_file():
        raise ValueError("native_service_environment_missing")
    env_text = environment.read_text(encoding="utf-8", errors="replace")
    host = next((line.split()[1] for line in env_text.splitlines()
                 if line.startswith("Linux ")), "")
    image_line = next((line for line in env_text.splitlines()
                       if line.startswith("image_id=")), "")
    env_image = ""
    for token in image_line.split():
        if token.startswith("image_id="):
            env_image = token.split("=", 1)[1]
    if not env_image or str(deployment_image_id) != env_image:
        raise ValueError("native_service_environment_image_mismatch")
    if not host or not host.lower().startswith(str(stack).lower()):
        raise ValueError("native_service_environment_stack_mismatch")
    if str(execution_core).lower() != "aicpu":
        raise ValueError("native_service_requires_aicpu_active_child")
    if str(placement) != "intra_node_ub":
        raise ValueError("native_service_requires_intra_node_ub")
    topology = path.parents[1] / "topology.txt"
    # Preserve missing vs declared-unknown vs a concrete value.  The native ABI
    # does not expose the deployed HCCL algorithm/protocol; do not turn that
    # into permission to migrate to any requested value.
    def _declared(value):
        text = str(value or "").strip()
        if not text:
            return None
        return "unknown" if text.lower().startswith("unknown") else text

    identity = {
        "path": str(path), "summary_sha256": _file_sha256(path),
        "manifest_path": str(manifest), "manifest_sha256": source_id,
        "versions_path": str(versions), "versions_sha256": _file_sha256(versions),
        "environment_path": str(environment),
        "environment_sha256": _file_sha256(environment),
        "environment_image_id": env_image, "host": host,
        "topology_path": str(topology),
        "topology_sha256": _file_sha256(topology) if topology.is_file() else None,
        "topology_interpretation": str(version.get("topology_interpretation") or ""),
        "scope": str(summary.get("scope") or ""),
        "python_packages": {key: packages.get(key)
                            for key in ("torch", "torch_npu", "cann") if packages.get(key)},
        "hccl_algorithm": _declared(version.get("hccl_algorithm")),
        "hccl_protocol": _declared(version.get("hccl_protocol")),
        "hccl_communicator_name": version.get("hccl_communicator_name"),
    }
    return summary, identity


def _native_service_conditions(summary, path, groups):
    """Build one condition per native target call, never summing ranks."""
    wanted = {int(group) for group in groups}
    conditions, heldout = [], set()
    for row in summary.get("conditions") or ():
        if int(row.get("group_size") or 0) not in wanted:
            continue
        if str(row.get("collective")) not in _NATIVE_LOCAL_COLLECTIVE_SET:
            continue
        if str(row.get("dtype") or "bf16").lower() not in {"bf16", "bfloat16"}:
            continue
        if str(row.get("status") or "ok") != "ok":
            raise ValueError("native_service_condition_status_not_ok")
        active, call_samples = [], []
        payload = int(row.get("input_bytes") or 0)
        rank_keys = sorted(int(key) for key in (row.get("per_rank") or {}))
        group = int(row.get("group_size") or 0)
        if rank_keys != [0, group - 1] or int(row.get("profile_rank_count") or 0) != 2:
            raise ValueError("native_service_rank_placement_not_declared")
        native_id = str(row.get("native_id") or
                        f"{row['collective']}:g{group}:{payload}")
        for rank_key, rank in sorted((int(key), value) for key, value in
                                     (row.get("per_rank") or {}).items()):
            if rank.get("canary_before_ok") is False or rank.get(
                    "canary_after_ok") is False or rank.get("mapping_ok") is False:
                raise ValueError("native_service_rank_evidence_failed")
            if rank.get("actual_bytes_match") is False or int(
                    rank.get("actual_input_bytes") or 0) != payload:
                raise ValueError("native_service_actual_input_mismatch")
            calls = rank.get("calls") or ()
            rank_values = []
            for call_index, call in enumerate(calls):
                ai_rows = call.get("ai_cpu_rows") or ()
                if not ai_rows or any(row.get("kind") != "ai_cpu"
                                      or "HcclLaunchAicpuKernel" not in str(
                                          row.get("name") or "")
                                      for row in ai_rows):
                    raise ValueError("native_service_active_aicpu_evidence_missing")
                value = call.get("ai_cpu_union_us")
                if value is None:
                    continue
                value = float(value)
                rep_index = int(call.get("rep_index", call_index))
                rank_values.append(value)
                call_samples.append({
                    "native_id": native_id, "rank": rank_key,
                    "repeat_index": rep_index,
                    "marker": call.get("marker"),
                    "range_marker": call.get("range_marker"),
                    "status": call.get("status"),
                    "duration_us": value,
                    "communication_union_us": call.get("communication_union_us"),
                    "overlap_us": call.get("overlap_us"),
                    "combined_span_union_us": call.get("combined_span_union_us"),
                    "range_duration_us": call.get("range_duration_us"),
                    "arrival_ns": call.get("arrival_ns"),
                    "enqueue_start_ns": call.get("enqueue_start_ns"),
                    "enqueue_end_ns": call.get("enqueue_end_ns"),
                    "complete_ns": call.get("complete_ns"),
                    "time_base_us": call.get("time_base_us"),
                    "span_start_rel_us": call.get("span_start_rel_us"),
                    "span_end_rel_us": call.get("span_end_rel_us"),
                    "ai_cpu_row_count": call.get("ai_cpu_row_count"),
                    "communication_row_count": call.get("communication_row_count"),
                })
            samples = rank.get("ai_cpu_union_us") or ()
            samples = [float(value) for value in samples if value is not None]
            if samples:
                active.append(median(samples))
            elif rank_values:
                active.append(median(rank_values))
        if not active or payload <= 0:
            continue
        split = str(row.get("split") or "train")
        if split == "heldout":
            heldout.add(payload)
        conditions.append({
            "native_id": native_id,
            "collective": row["collective"],
            "world_size": group,
            "tokens": row.get("tokens"),
            "hidden_size": row.get("hidden_size"),
            "payload_bytes": payload,
            "output_bytes": row.get("output_bytes"),
            "per_op_ms": median(active) / 1000.0,
            "dtype": "bf16", "placement_scope": "intra_node_ub",
            "split": split, "single": False,
            "run_id": native_id,
            "observed_expansion_mode": "aicpu",
            "requested_expansion_mode": "unspecified",
            "source": str(path),
            "measurement": {
                **_measurement_boundary(
                    row["collective"], "device_kernel", True),
                "explicit": True,
            },
            "transport_window": {
                "kind": "native_single_target_call", "phase": "aicpu_child",
                "warmup_calls": 3, "admitted_repeats": 5,
                "profiled_ranks": [0, group - 1],
                "clock_domain": "device", "concurrent_collectives": False,
            },
            "direction_observed": False,
            "algorithm_identified": False,
            "call_samples": call_samples,
            "aggregation": "median_of_rank_medians",
            "sample_source": "native_ai_cpu_child",
        })
    if not conditions:
        raise ValueError("native_service_no_compatible_conditions")
    return conditions, heldout


@lru_cache(maxsize=4)
def _collect_native_service_cached(source: str, groups: tuple, stack: str,
                                   cann: str, execution_core: str,
                                   placement: str, mtime_ns: int, size: int,
                                   deployment_image_id: str):
    """Fit the existing A5 ServiceCurve bundle from a validated native wave.

    The cached producer reads the source artifacts only on a cache miss.  It
    validates actual environment host/image, manifest SHA, CANN/topology,
    group/dtype/canary/mapping evidence and the requested deployment image.
    The fitted response is the active AICPU child, never the COMMUNICATION
    envelope or a sum of parent/alias.
    """
    path = Path(source)
    summary, _identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, heldout = _native_service_conditions(summary, path, groups)
    campaign = {
        "label": f"native-service:{str(summary.get('manifest_sha256'))[:12]}",
        "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "stack": str(stack), "cann": str(cann), "dtype": "bf16",
        "placement": placement, "observed_expansion_mode": execution_core,
        "requested_expansion_mode": "unspecified", "source": str(path),
        "conditions": conditions,
    }
    ingested = _ingest_a5_campaign_data(campaign)
    return fit_a5_service_curves(
        ingested, holdout_payloads=tuple(sorted(heldout)))


def collect_native_service(source: str, *, stack: str, cann: str,
                           execution_core: str, placement: str,
                           deployment_image_id: str, groups=(2, 4, 8)):
    """Public cached collector; only source revision/deployment and axes key it."""
    path = Path(source)
    stat = path.stat()
    return _collect_native_service_cached(
        str(path), tuple(int(group) for group in groups), str(stack),
        str(cann), str(execution_core), str(placement),
        int(stat.st_mtime_ns), int(stat.st_size), str(deployment_image_id))


_APPLICABILITY_KEYS = (
    ("platform", "family", "platform.family"),
    ("platform", "variant", "platform.variant"),
    ("platform", "sku", "platform.sku"),
    ("platform", "key", "platform.key"),
    ("platform", "arch_family", "platform.arch_family"),
    ("topology", "kind", "topology.kind"),
    ("topology", "scope", "topology.scope"),
    ("topology", "group_size", "topology.group_size"),
    ("topology", "host", "topology.host"),
    ("topology", "nodes", "topology.nodes"),
    ("topology", "nic", "topology.nic"),
    ("software", "stack", "software.stack"),
    ("software", "cann", "software.cann"),
    ("software", "execution_core", "software.execution_core"),
    ("software", "torch", "software.torch"),
    ("software", "torch_npu", "software.torch_npu"),
    ("software", "hccl_algorithm", "software.hccl_algorithm"),
    ("software", "hccl_protocol", "software.hccl_protocol"),
    ("transport_window", "kind", "transport_window.kind"),
    ("transport_window", "phase", "transport_window.phase"),
    ("transport_window", "warmup_calls", "transport_window.warmup_calls"),
    ("transport_window", "admitted_repeats", "transport_window.admitted_repeats"),
    ("transport_window", "concurrent_collectives",
     "transport_window.concurrent_collectives"),
    ("units", "work", "units.work"),
    ("units", "duration", "units.duration"),
    ("units", "rate", "units.rate"),
    ("units", "rate_scope", "units.rate_scope"),
    ("units", "active_units", "units.active_units"),
    ("units", "dtype", "units.dtype"),
    ("units", "service_work_unit", "units.service_work_unit"),
    ("shared_domain", "id", "shared_domain.id"),
    ("shared_domain", "kind", "shared_domain.kind"),
    ("shared_domain", "scope", "shared_domain.scope"),
    ("shared_domain", "members", "shared_domain.members"),
    ("shared_domain", "host", "shared_domain.host"),
    ("shared_domain", "inter_node", "shared_domain.inter_node"),
    ("shared_domain", "nic", "shared_domain.nic"),
)
_APPLICABILITY_KEY_SET = frozenset(key for _, _, key in _APPLICABILITY_KEYS) | {
    "direction", "hardware"}
_LINEAGE_KEYS = (
    ("source", "summary_sha256", "source.summary_sha256"),
    ("source", "manifest_sha256", "source.manifest_sha256"),
    ("source", "versions_sha256", "source.versions_sha256"),
    ("source", "environment_image_id", "source.environment_image_id"),
)
_LINEAGE_KEY_SET = frozenset(key for _, _, key in _LINEAGE_KEYS)


def _flatten_sections(mapping, paths):
    flat = {}
    for outer, inner, flat_key in paths:
        section = mapping.get(outer)
        if isinstance(section, Mapping) and inner in section:
            flat[flat_key] = section[inner]
    return flat


def _applicability_sections(domain):
    if "conditions" in domain and isinstance(domain["conditions"], Mapping):
        return domain["conditions"]
    if "applicability" in domain and isinstance(domain["applicability"], Mapping):
        return domain["applicability"]
    if any(isinstance(domain.get(outer), Mapping)
           for outer, _, _ in _APPLICABILITY_KEYS):
        return domain
    return None


def _complete_applicability_identity(flat):
    """Add scalar hardware and reject a hardware/platform identity conflict."""
    hardware = flat.get("hardware")
    platform_key = flat.get("platform.key")
    platform_sku = flat.get("platform.sku")
    expected = platform_key or platform_sku
    if hardware is not None and expected is not None and str(hardware) != str(expected):
        raise ValueError("a5_local_service_hardware_identity_conflict")
    return flat


def a5_local_service_applicability_key(domain):
    """Canonical target-applicability key; source lineage is separate."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "applicability_key" in domain:
        return _complete_applicability_identity(
            {str(key): value for key, value in
             dict(domain["applicability_key"]).items()})
    conditions = _applicability_sections(domain)
    if conditions is not None:
        flat = _flatten_sections(conditions, _APPLICABILITY_KEYS)
        if "hardware" in conditions:
            flat["hardware"] = conditions["hardware"]
        if "direction" in conditions:
            flat["direction"] = conditions["direction"]
        return _complete_applicability_identity(flat)
    if not any(isinstance(value, Mapping) for value in domain.values()):
        flat = {str(key): value for key, value in domain.items()
                if key in _APPLICABILITY_KEY_SET}
        return _complete_applicability_identity(flat)
    return {}


def a5_local_service_lineage_key(domain):
    """Canonical calibration-source lineage, not target applicability."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "lineage_key" in domain:
        return {str(key): value for key, value in dict(domain["lineage_key"]).items()}
    source = domain.get("source")
    if isinstance(source, Mapping):
        flat = _flatten_sections({"source": source}, _LINEAGE_KEYS)
        if flat:
            return flat
        return {key: source.get(inner) for outer, inner, key in _LINEAGE_KEYS
                if outer == "source" and inner in source}
    if not any(isinstance(value, Mapping) for value in domain.values()):
        return {str(key): value for key, value in domain.items()
                if key in _LINEAGE_KEY_SET}
    return {}


def a5_local_service_domain_key(domain):
    """Backward-compatible merged key; require() checks the two parts apart."""
    return {**a5_local_service_applicability_key(domain),
            **a5_local_service_lineage_key(domain)}


def _declared_applicability(record):
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    declared_app = dict(record.get("applicability_key") or {})
    declared_lineage = dict(record.get("lineage_key") or {})
    if not declared_app and not declared_lineage:
        declared = dict(record.get("domain_key") or {})
        declared_app = {key: value for key, value in declared.items()
                        if key in _APPLICABILITY_KEY_SET}
        declared_lineage = {key: value for key, value in declared.items()
                            if key in _LINEAGE_KEY_SET}
    if not declared_app and not declared_lineage:
        raise ValueError("a5_local_service_record_requires_declared_domain")
    return declared_app, declared_lineage


def _presence_mismatches(declared, requested):
    missing = sorted(key for key in declared if key not in requested)
    changed = sorted(key for key in declared if key in requested
                     and requested[key] != declared[key])
    return missing, changed


def require_a5_local_service_applicability(record, domain):
    """Match the target environment only; calibration lineage is separate.

    This is the local counterpart of T06 ``service_conditions_match``:
    declared paths must be present and equal; extra requested paths and
    source lineage are not applicability.
    """
    declared_app, _declared_lineage = _declared_applicability(record)
    requested_app = a5_local_service_applicability_key(domain)
    missing, changed = _presence_mismatches(declared_app, requested_app)
    if missing or changed:
        raise ModelingError(
            "a5_local_service_domain_mismatch:applicability:"
            + ",".join(missing + changed))
    return declared_app


def require_a5_local_service_domain(record, domain):
    """Reject a reuse/supply respecification outside the measured domain.

    Applicability (target environment) and lineage (calibration source) are
    compared separately so a source hash cannot stand in for a changed runtime
    condition, and a shared source cannot hide a changed torch/CANN/algorithm.
    """
    declared_app, declared_lineage = _declared_applicability(record)
    require_a5_local_service_applicability(record, domain)
    requested_lineage = a5_local_service_lineage_key(domain)
    missing_lineage, changed_lineage = _presence_mismatches(
        declared_lineage, requested_lineage)
    if missing_lineage or changed_lineage:
        raise ModelingError(
            "a5_local_service_domain_mismatch:lineage:"
            + ",".join(missing_lineage + changed_lineage))
    return {**declared_app, **declared_lineage}


def _native_local_domain(source_identity, *, group_size, stack, cann,
                         direction, dtype):
    software_versions = source_identity.get("python_packages") or {}
    shared = {
        "id": f"ub_mesh:{source_identity['host']}:W{int(group_size)}",
        "kind": "ub_mesh", "scope": "intra_node",
        "members": int(group_size), "host": source_identity["host"],
        "inter_node": False, "nic": "na",
    }
    window = {
        "kind": "native_single_target_call", "phase": "aicpu_child",
        "warmup_calls": 3, "admitted_repeats": 5,
        "profiled_ranks": [0, int(group_size) - 1],
        "clock_domain": "device", "concurrent_collectives": False,
        "start": "unique hccl target-call marker/range",
        "end": "target-call completion",
    }
    conditions = {
        # ``hardware`` stays the T06/profile hardware key string; the
        # structured identity lives under ``platform`` so conditions can be
        # consumed by service_conditions_match without a key/type conflict.
        "hardware": A5_HARDWARE_KEY,
        "platform": {"family": A5_ARCH_FAMILY, "variant": "DT",
                     "sku": A5_SKU, "key": A5_HARDWARE_KEY,
                     "arch_family": A5_ARCH_FAMILY},
        "topology": {"kind": "ub_mesh", "scope": "intra_node",
                     "group_size": int(group_size),
                     "host": source_identity["host"], "nodes": 1, "nic": "na"},
        "software": {
            "stack": str(stack), "cann": str(cann),
            "execution_core": "aicpu",
            "torch": software_versions.get("torch"),
            "torch_npu": software_versions.get("torch_npu"),
            "hccl_algorithm": source_identity.get("hccl_algorithm"),
            "hccl_protocol": source_identity.get("hccl_protocol"),
        },
        "transport_window": window,
        "units": {"work": "bytes_per_rank_input", "duration": "ms",
                  "rate": "bytes_per_second", "rate_scope": "per_owner",
                  "active_units": int(group_size), "dtype": dtype,
                  "service_work_unit": "bytes"},
        "direction": direction,
        "shared_domain": shared,
    }
    measurement = {
        "kind": "native_target_call", "phase": "aicpu_child",
        "resource": "aicpu_child",
        "work_coordinate": "per_rank_input_bytes", "duration_unit": "ms",
        "source_kind": "native_ai_cpu_child",
        "service_semantics": "aicpu_child_collective_phase",
        "complete_collective_call": False, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": False, "direction_observed": False,
    }
    domain = {
        **conditions,
        "source": source_identity,
        "measurement_boundary": measurement,
        "provenance": {
            "direction_basis": "hccl_source_template_native_direction",
            "direction_observed": False,
            "algorithm_identified": False,
            "shared_domain_basis": "declared_intrahost_ub_mesh_and_group_size",
            "transport_window_basis": "native_target_call_manifest",
        },
    }
    applicability_key = a5_local_service_applicability_key(conditions)
    lineage_key = a5_local_service_lineage_key({"source": source_identity})
    domain_key = {**applicability_key, **lineage_key}
    return domain, domain_key, applicability_key, lineage_key


def _local_service_evidence_gaps(collective, group_size):
    return [
        "link_transit_is_not_separated_from_the_aicpu_child_phase",
        "per_hop_link_price_is_not_identified",
        f"measured_only:{collective}:W{int(group_size)}:bf16:aicpu:intrahost:rack08:cann9.2.0",
        "cross_node_aiv_and_other_skus_are_unmeasured",
        "pr_variant_is_unmeasured",
        "hccl_algorithm_and_protocol_are_not_exposed_by_the_python_abi",
        "heldout_split_is_by_payload_not_by_environment",
    ]


@lru_cache(maxsize=8)
def _fit_native_local_service_cached(source: str, collective: str, group_size: int,
                                     stack: str, cann: str, execution_core: str,
                                     placement: str, deployment_image_id: str,
                                     dtype: str, mtime_ns: int, size: int):
    path = Path(source)
    summary, source_identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, heldout = _native_service_conditions(
        summary, path, (int(group_size),))
    rows = [row for row in conditions if row["collective"] == collective]
    if not rows:
        raise ValueError(f"native_local_service_collective_missing:{collective}")
    direction = _NATIVE_LOCAL_DIRECTION[collective]
    domain, domain_key, applicability_key, lineage_key = _native_local_domain(
        source_identity, group_size=int(group_size),
        stack=stack, cann=cann, direction=direction, dtype=dtype)
    gaps = _local_service_evidence_gaps(collective, int(group_size))
    samples, source_records = [], []
    for row in rows:
        calls = row.get("call_samples") or ()
        if not calls:
            samples.append({
                "work": int(row["payload_bytes"]),
                "duration_us": float(row["per_op_ms"]) * 1000.0,
                "run_id": str(row["run_id"]), "repeat_index": 0,
                "split": str(row["split"]),
                "source": str(row.get("source") or path)})
            source_records.append({
                "native_id": row["native_id"], "rank": None,
                "repeat_index": None, "marker": None,
                "status": "aggregate_only", "used_in_fit": True,
                "work": int(row["payload_bytes"]), "split": str(row["split"]),
                "source": str(row.get("source") or path)})
            continue
        for call in calls:
            ok = call.get("status") in (None, "ok")
            record = {
                "native_id": row["native_id"], "rank": call["rank"],
                "repeat_index": call["repeat_index"],
                "marker": call.get("marker"),
                "range_marker": call.get("range_marker"),
                "status": call.get("status"), "used_in_fit": ok,
                "work": int(row["payload_bytes"]), "split": str(row["split"]),
                "source": str(row.get("source") or path),
            }
            source_records.append(record)
            if not ok:
                continue
            samples.append({
                "work": int(row["payload_bytes"]),
                "duration_us": float(call["duration_us"]),
                "run_id": f"{row['native_id']}:rank{call['rank']}",
                "repeat_index": int(call["repeat_index"]),
                "split": str(row["split"]),
                "source": (f"{path}:{row['native_id']}:rank{call['rank']}:"
                           f"rep{call['repeat_index']}:{call.get('marker')}"),
            })
    if not samples:
        raise ValueError("native_local_service_no_training_or_heldout_samples")
    packet = {
        "id": (f"a5-native-aicpu-child:{collective}:W{int(group_size)}:"
               f"{stack}:{cann}:{placement}"),
        "scope": "independent_resource",
        "resources": ["aicpu_child"],
        "phase": "collective_expand",
        "work_unit": "bytes",
        "boundary": "resource_phase",
        "active_units": int(group_size),
        "rate_scope": "per_owner",
        "conditions": dict(applicability_key),
        "identification": {
            "status": "declared_effective_service",
            "resource_identified": False,
            "concurrency_scope": "isolated_invocation",
        },
        "provenance": {
            "source": str(path), "sources": [str(path)],
            "window": "single_target_call",
            "observation_kind": "native_ai_cpu_child",
            "boundary_kind": "resource_phase", "scope": "independent",
            "service_semantics": "aicpu_child_collective_phase",
            "measurement_boundary": domain["measurement_boundary"],
            "direction": direction,
            "shared_domain": domain["shared_domain"]["id"],
            "transport_window": domain["transport_window"],
            "rate_interpretation": (
                "effective_aicpu_child_phase_rate_not_an_isolated_aicpu_"
                "or_link_peak"),
            "source_identity": source_identity,
            "source_records": source_records,
            "aggregation": "median_of_run_medians_per_rank",
            "sample_source": "native_ai_cpu_child",
            "evidence_gaps": gaps,
        },
        "samples": samples,
    }
    fitted = fit_resource_service(packet)
    heldout_error = max((abs(float(row["relative_error"]))
                         for row in fitted["heldout"]), default=None)
    if heldout_error is not None and heldout_error > 0.20:
        gaps = gaps + [
            f"phase_rate_weakly_identified:heldout_max_relative_error="
            f"{heldout_error:.3f}"]
    domain = {**domain, "evidence_gaps": list(gaps)}
    all_calls = [call for row in rows for call in (row.get("call_samples") or ())]
    child_overlapped = bool(all_calls) and all(
        call.get("overlap_us") is not None
        and call.get("combined_span_union_us") is not None
        and call.get("communication_union_us") is not None
        and abs(float(call["duration_us"]) - float(call["overlap_us"])) < 1e-9
        and abs(float(call["combined_span_union_us"])
                - float(call["communication_union_us"])) < 1e-9
        for call in all_calls)
    return {
        "schema": A5_COMM_LOCAL_SERVICE_SCHEMA,
        "diagnostic_only": True,
        "production_eligible": False,
        "overlap_with_communication": child_overlapped,
        "child_is_complete_collective": False,
        "service": fitted["service"],
        "normalized_schema": fitted["service"].get("schema"),
        "formula": fitted["formula"],
        "model_id": fitted["model_id"],
        "fit": fitted["fit"],
        "heldout": fitted["heldout"],
        "heldout_max_abs_relative_error": heldout_error,
        "domain": domain,
        "domain_key": domain_key,
        "applicability_key": applicability_key,
        "lineage_key": lineage_key,
        "service_semantics": "aicpu_child_collective_phase",
        "evidence_gaps": gaps,
        "source": source_identity,
        "source_records": source_records,
        "holdout_payloads": sorted(heldout),
    }


def fit_a5_native_local_service(source: str = NATIVE_A5_SERVICE_SOURCE, *,
                                collective: str, stack: str, cann: str,
                                deployment_image_id: str, group_size: int = 2,
                                execution_core: str = "aicpu",
                                placement: str = "intra_node_ub",
                                dtype: str = "bf16"):
    """Fit one T06 normalized local A5 service from native AI_CPU evidence.

    The service is the directly observed ``HcclLaunchAicpuKernel`` AI_CPU child
    phase of one target call.  It is a phase service, not an isolated AICPU
    hardware peak and not the COMMUNICATION envelope; the link transit and
    per-hop price remain evidence gaps.  The work axis is the captured per-rank
    input payload, so no hop factor is invented here.
    """
    name = _collective(collective)
    if name not in _NATIVE_LOCAL_COLLECTIVES:
        raise ValueError(f"native_local_service_collective_unsupported:{name}")
    if _dtype(dtype) != "bf16":
        raise ValueError("native_local_service_requires_bf16")
    if int(group_size) < 2:
        raise ValueError("native_local_service_requires_group_size_at_least_two")
    path = Path(source)
    stat = path.stat()
    return _fit_native_local_service_cached(
        str(path), name, int(group_size), str(stack), str(cann),
        str(execution_core), str(placement), str(deployment_image_id),
        "bf16", int(stat.st_mtime_ns), int(stat.st_size))


def native_child_stage_evidence(source: str = NATIVE_A5_SERVICE_SOURCE, *,
                                collective: str, stack: str, cann: str,
                                deployment_image_id: str, group_size: int = 2,
                                execution_core: str = "aicpu",
                                placement: str = "intra_node_ub", split=None,
                                used_for_tuning=()):
    """Per-call native child stage/overlap evidence; no rank/rep aggregation.

    This is a diagnostic/handoff source, not a prediction lookup.  Rows used
    to explain or tune the model keep their original split/role and are marked
    ``used_for_tuning``; they are not presented as independent generalization.
    """
    name = _collective(collective)
    if name not in _NATIVE_LOCAL_COLLECTIVES:
        raise ValueError(f"native_local_service_collective_unsupported:{name}")
    path = Path(source)
    summary, source_identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, _heldout = _native_service_conditions(
        summary, path, (int(group_size),))
    service_record = fit_a5_native_local_service(
        source, collective=name, stack=stack, cann=cann,
        deployment_image_id=deployment_image_id, group_size=group_size,
        execution_core=execution_core, placement=placement)
    service = service_record["service"]
    fixed_us = float(service["fixed_us"])
    rate = float(service["effective_rate"])
    tuning = set(str(item) for item in (used_for_tuning or ()))
    records = []
    for condition in conditions:
        if condition["collective"] != name:
            continue
        if split is not None and condition["split"] != split:
            continue
        work = int(condition["payload_bytes"])
        predicted_us = fixed_us + work * 1e6 / rate
        for call in condition.get("call_samples") or ():
            if call.get("status") not in (None, "ok"):
                continue
            child = float(call["duration_us"])
            communication = call.get("communication_union_us")
            overlap = call.get("overlap_us")
            combined = call.get("combined_span_union_us")
            communication = None if communication is None else float(communication)
            overlap = None if overlap is None else float(overlap)
            combined = None if combined is None else float(combined)
            used = condition["native_id"] in tuning
            records.append({
                "native_id": condition["native_id"],
                "collective": condition["collective"],
                "tokens": condition.get("tokens"),
                "hidden_size": condition.get("hidden_size"),
                "work_bytes": work,
                "original_split": condition["split"],
                "evidence_role": (condition["split"]
                                  + ("_used_for_tuning" if used else "")),
                "used_for_tuning": used,
                "rank": call["rank"],
                "repeat_index": call["repeat_index"],
                "marker": call.get("marker"),
                "range_marker": call.get("range_marker"),
                "child_us": child,
                "communication_union_us": communication,
                "overlap_us": overlap,
                "combined_span_union_us": combined,
                "range_duration_us": call.get("range_duration_us"),
                "arrival_ns": call.get("arrival_ns"),
                "enqueue_start_ns": call.get("enqueue_start_ns"),
                "enqueue_end_ns": call.get("enqueue_end_ns"),
                "complete_ns": call.get("complete_ns"),
                "child_fraction_of_communication": (
                    None if not communication else child / communication),
                "overlap_fraction_of_child": (
                    None if not child or overlap is None else overlap / child),
                "child_within_communication": bool(
                    overlap is not None and communication is not None
                    and combined is not None
                    and abs(child - overlap) < 1e-9
                    and abs(combined - communication) < 1e-9),
                "predicted_us": predicted_us,
                "relative_error": predicted_us / child - 1,
                "source": str(path),
            })
    if not records:
        raise ValueError("native_child_evidence_no_records")
    split_counts = {}
    for record in records:
        split_counts[record["original_split"]] = split_counts.get(
            record["original_split"], 0) + 1
    return {
        "schema": A5_NATIVE_CHILD_EVIDENCE_SCHEMA,
        "source": source_identity,
        "collective": name,
        "group_size": int(group_size),
        "aggregation": "per_call_no_cross_rank_median",
        "service_model_id": service.get("model_id") or service_record.get("model_id"),
        "service_formula": service_record["formula"],
        "service_diagnostic_only": service_record.get("diagnostic_only"),
        "service_production_eligible": service_record.get("production_eligible"),
        "records": records,
        "summary": {
            "count": len(records),
            "split_counts": split_counts,
            "used_for_tuning_native_ids": sorted(tuning),
            "used_for_tuning_count": sum(
                1 for record in records if record["used_for_tuning"]),
            "all_child_within_communication": all(
                record["child_within_communication"] for record in records),
            "service_fixed_us": fixed_us,
            "service_effective_rate": rate,
            "child_min_us": min(record["child_us"] for record in records),
            "child_max_us": max(record["child_us"] for record in records),
            "fixed_term_exceeds_some_child_calls": any(
                record["child_us"] < fixed_us for record in records),
            "child_by_work": {
                str(work): {
                    "count": sum(1 for record in records
                                 if record["work_bytes"] == work),
                    "min_us": min(
                        (record["child_us"] for record in records
                         if record["work_bytes"] == work), default=None),
                    "max_us": max(
                        (record["child_us"] for record in records
                         if record["work_bytes"] == work), default=None),
                }
                for work in sorted({record["work_bytes"] for record in records})
            },
            "warning": ("used-for-tuning rows keep their original split and "
                        "are not independent generalization"),
        },
    }


def _normalized_local_service(record):
    if isinstance(record, Mapping) and record.get("schema") == A5_COMM_LOCAL_SERVICE_SCHEMA:
        return normalize_resource_service(record["service"])
    return normalize_resource_service(record)


def _require_local_work_unit(service, work_unit):
    if work_unit is not None and str(work_unit) != service.work_unit:
        raise ValueError("a5_local_service_work_unit_mismatch")


def a5_local_service_prediction(record, work, *, domain=None, work_unit=None,
                                work_scope=None):
    """Evaluate the ordinary T06 formula, optionally after a domain/unit check."""
    service = _normalized_local_service(record)
    _require_local_work_unit(service, work_unit)
    if domain is not None:
        require_a5_local_service_applicability(
            record if isinstance(record, Mapping) else {}, domain)
    return service.prediction(work, work_scope=work_scope)


def a5_local_service_supply_change(record, work, *, effective_rate, domain,
                                   work_unit=None):
    """Change the declared effective phase supply only inside its domain.

    ``domain`` is required: an A5 supply update that omits the measured
    environment would be a cross-domain reuse by construction.  Generic T06
    callers that really want an unscoped rate intervention use
    ``NormalizedResourceService.counterfactual_effective_rate`` directly.
    """
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    require_a5_local_service_domain(record, domain)
    service = _normalized_local_service(record)
    _require_local_work_unit(service, work_unit)
    changed = service.counterfactual_effective_rate(work, effective_rate=effective_rate)
    return {
        **changed,
        "domain_checked": True,
        "domain_key": dict(record.get("domain_key") or {}),
        "source_model_id": service.model_id,
        "evidence_gaps": list(record.get("evidence_gaps") or ()),
    }
