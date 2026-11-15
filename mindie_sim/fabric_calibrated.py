"""Mode B calibrated piecewise fabric profiles.

[Role]
  Price collectives from measured ``(payload, alpha, beta)`` points, with typed
  range/extrapolation policy and declared fit-condition matching.

[Invariants]
  * Interpolation from a zero-byte anchor is linear in ``size_bytes``; it does
    not clamp every interior payload to the left point.
  * ``endpoints`` and every keyword used by the EP knee/xPyD callers are
    accepted; fit conditions on ``world_size``, ``collective``, ``dtype`` and
    ``topology`` are enforced.
  * A calibrated curve is a uniform-traffic fit.  A skewed
    ``payloads_by_destination`` vector it did not declare support for fails
    closed (``route_qualified=False``, evidence
    ``calibrated_profile_uniform_only``) instead of reusing a uniform 0.75-us
    number for a 3,000-B receiver incast.
  * An unmeasured point disqualifies the route; extrapolation is explicit.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable records; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.fabric_topology import (COLLECTIVES, CollectiveTiming,
                                        FabricRangeError, FabricTier, RankEndpoint,
                                        _EXTRAPOLATION_POLICIES,
                                        _coerce_endpoints, _count_hops,
                                        _effective_bw_gbps, _positive)
from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class ProbePoint:
    payload_bytes: float
    alpha_us: float
    beta_ns_per_byte: float
    congestion_factor: float = 1.0
    load_basis: str = "isolated"
    extrapolated: bool = False
    fit_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_bytes", _positive(
            "probe_payload_bytes", self.payload_bytes, strict=False))
        object.__setattr__(self, "alpha_us", _positive(
            "probe_alpha_us", self.alpha_us, strict=False))
        object.__setattr__(self, "beta_ns_per_byte", _positive(
            "probe_beta", self.beta_ns_per_byte, strict=False))
        object.__setattr__(self, "congestion_factor", _positive(
            "probe_congestion", self.congestion_factor, minimum=1.0, strict=False))
        if self.load_basis not in {"isolated", "effective_under_load", "unmeasured"}:
            raise ModelingError(f"fabric_probe_load_basis_invalid:{self.load_basis}")
        if self.load_basis == "effective_under_load" and float(self.congestion_factor) != 1.0:
            raise ModelingError("fabric_probe_effective_load_double_congestion")
        if self.fit_confidence is not None and not 0.0 <= float(self.fit_confidence) <= 1.0:
            raise ModelingError(f"fabric_probe_confidence_invalid:{self.fit_confidence}")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

@dataclass(frozen=True)
class CalibratedTier:
    name: str
    domain_ranks: int
    points: tuple[ProbePoint, ...]
    oversubscription: float = 1.0
    fit_conditions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        FabricTier(self.name, self.domain_ranks, 0.0, 1.0)
        ordered = tuple(sorted(self.points, key=lambda point: point.payload_bytes))
        if not ordered:
            raise ModelingError(f"fabric_calibrated_points_missing:{self.name}")
        if len({point.payload_bytes for point in ordered}) != len(ordered):
            raise ModelingError(f"fabric_calibrated_payload_duplicate:{self.name}")
        if any(point.load_basis == "effective_under_load" for point in ordered) \
                and float(self.oversubscription) != 1.0:
            raise ModelingError("fabric_calibrated_effective_load_oversubscription")
        object.__setattr__(self, "points", ordered)
        object.__setattr__(self, "oversubscription", _positive(
            "oversubscription", self.oversubscription, minimum=1.0, strict=False))

    def at(self, payload: float, *, policy: str = "out_of_range") -> ProbePoint:
        if policy not in _EXTRAPOLATION_POLICIES:
            raise ModelingError(f"fabric_extrapolation_policy_invalid:{policy}")
        value = _positive("probe_payload", payload, strict=False)
        points = self.points
        low, high = points[0].payload_bytes, points[-1].payload_bytes
        if value < low or value > high:
            if policy != "extrapolate":
                raise FabricRangeError(f"fabric_probe_out_of_range:{self.name}:{value}")
            if value <= 0.0:
                raise FabricRangeError(f"fabric_probe_out_of_range:{self.name}:{value}")
            lo, hi = (points[0], points[min(1, len(points) - 1)]) if value < low else (
                points[max(0, len(points) - 2)], points[-1])
        elif value == low:
            return replace(points[0], extrapolated=False)
        elif value == high:
            return replace(points[-1], extrapolated=False)
        else:
            index = next(i for i, point in enumerate(points) if point.payload_bytes >= value)
            lo, hi = points[index - 1], points[index]
        if hi.payload_bytes == lo.payload_bytes:
            return replace(lo, extrapolated=value < low or value > high)
        if lo.payload_bytes <= 0.0 or hi.payload_bytes <= 0.0 or value <= 0.0:
            # Linear in size_bytes from the (0 B, alpha0, beta0) anchor.  The
            # previous branch clamped every interior payload to the left point.
            low_value = max(0.0, float(lo.payload_bytes))
            high_value = max(low_value + 1e-12, float(hi.payload_bytes))
            fraction = min(1.0, max(0.0, (value - low_value)
                                   / (high_value - low_value)))
        else:
            fraction = ((math.log2(value) - math.log2(lo.payload_bytes))
                        / (math.log2(hi.payload_bytes) - math.log2(lo.payload_bytes)))
        blend = lambda a, b: float(a) + (float(b) - float(a)) * fraction
        return ProbePoint(value, blend(lo.alpha_us, hi.alpha_us),
                          blend(lo.beta_ns_per_byte, hi.beta_ns_per_byte),
                          blend(lo.congestion_factor, hi.congestion_factor),
                          (lo.load_basis if lo.load_basis == hi.load_basis else
                           "unmeasured" if "unmeasured" in {lo.load_basis, hi.load_basis}
                           else "isolated"),
                          value < low or value > high,
                          min(lo.fit_confidence if lo.fit_confidence is not None else 1.0,
                              hi.fit_confidence if hi.fit_confidence is not None else 1.0))

    def fit_mismatch(self, *, world_size: int, collective: str,
                     dtype: str | None = None,
                     topology: str | None = None) -> str | None:
        """Return the first unmet declared fit condition, if any."""
        for key, raw in self.fit_conditions:
            value = str(raw).strip()
            if key in {"world_size", "ranks"}:
                if int(world_size) != int(value):
                    return f"world_size:{world_size}!={value}"
            elif key == "min_world_size":
                if int(world_size) < int(value):
                    return f"min_world_size:{world_size}<{value}"
            elif key == "max_world_size":
                if int(world_size) > int(value):
                    return f"max_world_size:{world_size}>{value}"
            elif key == "collective":
                allowed = {item.strip().lower() for item in value.split(",") if item.strip()}
                if str(collective).strip().lower() not in allowed:
                    return f"collective:{collective}!={value}"
            elif key in {"dtype", "kv_dtype", "activation_dtype"}:
                if dtype is None or str(dtype).strip().lower() != value.lower():
                    return f"dtype:{dtype}!={value}"
            elif key in {"topology", "topology_preset"}:
                if topology is None or str(topology).strip().lower() != value.lower():
                    return f"topology:{topology}!={value}"
            else:
                return f"unsupported_fit_condition:{key}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "domain_ranks": self.domain_ranks,
                "oversubscription": self.oversubscription,
                "fit_conditions": dict(self.fit_conditions),
                "points": [dict(vars(point)) for point in self.points]}

@dataclass(frozen=True)
class CalibratedFabricProfile:
    """Mode B: piecewise measured fabric curves, no cluster-name assumptions."""

    tiers: tuple[CalibratedTier, ...]
    source: str = "fabric-probe"
    probe_seconds: float = 0.0
    assumptions: tuple[tuple[str, str], ...] = ()
    #: Explicit declaration that the fit covers skewed destination payloads.
    skew_supported: bool = False

    def __post_init__(self) -> None:
        domains = [tier.domain_ranks for tier in self.tiers]
        if not domains:
            raise ModelingError("calibrated_fabric_requires_tiers")
        if domains != sorted(set(domains)):
            raise ModelingError(f"calibrated_tier_domains_not_increasing:{domains}")
        if len({tier.name for tier in self.tiers}) != len(self.tiers):
            raise ModelingError("calibrated_tier_names_not_unique")

    @classmethod
    def from_probe(cls, payload: Mapping[str, Any]) -> "CalibratedFabricProfile":
        if not isinstance(payload, Mapping):
            raise ModelingError("fabric_probe_mapping_required")
        raw_tiers = payload.get("tiers")
        if raw_tiers is None and payload.get("measurements") is not None:
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in payload["measurements"]:
                if not isinstance(row, Mapping) or not row.get("tier"):
                    raise ModelingError("fabric_probe_tier_evidence_missing")
                grouped.setdefault(str(row["tier"]), []).append(row)
            raw_tiers = [{"name": name,
                          "domain_ranks": max(int(row.get("ranks", 2) or 2)
                                              for row in rows),
                          "points": rows} for name, rows in grouped.items()]
        if not isinstance(raw_tiers, Sequence) or isinstance(raw_tiers, (str, bytes)):
            raise ModelingError("fabric_probe_tiers_invalid")
        tiers = []
        for raw in raw_tiers:
            if not isinstance(raw, Mapping):
                raise ModelingError("fabric_probe_tier_invalid")
            points = tuple(ProbePoint(
                payload_bytes=row.get("payload_bytes", row.get("bytes")),
                alpha_us=row.get("alpha_us"), beta_ns_per_byte=row.get("beta_ns_per_byte"),
                congestion_factor=row.get("congestion_factor", 1.0),
                load_basis=str(row.get("load_basis", "isolated")),
                fit_confidence=row.get("fit_confidence"))
                for row in raw.get("points", ()))
            tiers.append(CalibratedTier(
                str(raw.get("name")), int(raw.get("domain_ranks")), points,
                float(raw.get("oversubscription", 1.0) or 1.0),
                tuple(sorted((str(k), str(v)) for k, v in
                             dict(raw.get("fit_conditions") or {}).items()))))
        assumptions = dict(payload.get("assumptions") or {})
        skew_supported = bool(
            payload.get("skew_supported", False)
            or str(assumptions.get("skew_supported", "")).strip().lower()
            in {"1", "true", "yes", "declared"})
        return cls(tuple(tiers), str(payload.get("source", "fabric-probe")),
                   float(payload.get("probe_seconds", 0.0) or 0.0),
                   tuple(sorted((str(k), str(v)) for k, v in assumptions.items())),
                   skew_supported)

    def _tier(self, name: str) -> CalibratedTier:
        return next(tier for tier in self.tiers if tier.name == name)

    def peer_transfer(self, payload_bytes: float,
                      endpoints: Sequence[int | RankEndpoint], *,
                      dtype: str | None = None,
                      topology: str | None = None) -> CollectiveTiming:
        """Price a point-to-point hop from the measured piecewise curves.

        The measured fit is per traversed tier.  An unmeasured point makes the
        actual payload route unqualified; extrapolation policy and fit
        conditions are enforced exactly as for a collective.
        """
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        if endpoints is not None:
            endpoints = _coerce_endpoints(list(endpoints))
        ranks = len(endpoints or ())
        if ranks < 2 or payload <= 0.0:
            return CollectiveTiming("peer_transfer", max(1, ranks), payload,
                                    0.0, (), "none", 0.0, 0.0, 0.0, "calibrated_peer",
                                    (), (), True, "trivial")
        shells = [FabricTier(tier.name, tier.domain_ranks, 0.0, 1.0,
                             oversubscription=tier.oversubscription,
                             evidence="measured_profile") for tier in self.tiers]
        hops = _count_hops(shells, ranks)
        per_tier: list[tuple[str, float]] = []
        unmeasured = False
        total = 0.0
        for shell, groups, _before, _after in hops:
            tier = self._tier(shell.name)
            mismatch = tier.fit_mismatch(
                world_size=int(ranks), collective="peer_transfer",
                dtype=dtype, topology=topology)
            if mismatch is not None:
                raise ModelingError(
                    f"calibrated_fit_condition_mismatch:{shell.name}:{mismatch}")
            measured = tier.at(payload)
            if measured.load_basis == "unmeasured":
                unmeasured = True
            os_factor = (shell.oversubscription
                         if shell.semantics in {"shared", "mesh"} else 1.0)
            us = (measured.alpha_us * max(1, groups - 1)
                  + payload * measured.beta_ns_per_byte
                  * measured.congestion_factor * os_factor / 1000.0)
            per_tier.append((shell.name, us))
            total += us
        bottleneck = max(per_tier, key=lambda item: item[1])
        return CollectiveTiming(
            "peer_transfer", ranks, payload, total, tuple(per_tier),
            bottleneck[0], float(bottleneck[1]), payload, 0.0,
            "calibrated_peer_point_to_point", (), tuple(name for name, _ in per_tier),
            not unmeasured, "unmeasured_profile" if unmeasured else "measured_profile",
            max_sender_egress_bytes=payload,
            max_receiver_ingress_bytes=payload, cut_bytes=payload)

    def peer_resources(self, payload_bytes: float,
                       endpoints: Sequence[int | RankEndpoint], *,
                       dtype: str | None = None,
                       topology: str | None = None,
                       ) -> tuple[tuple[str, float], ...]:
        """Measured per-tier resource services for one point-to-point hop."""
        timing = self.peer_transfer(payload_bytes, endpoints, dtype=dtype,
                                    topology=topology)
        return tuple((f"tier:{name}", float(us))
                     for name, us in timing.tier_us if float(us) > 0.0)

    def peer_cut_demands(self, payload_bytes: float,
                         endpoints: Sequence[int | RankEndpoint], *,
                         dtype: str | None = None,
                         topology: str | None = None) -> dict[str, float]:
        """Bytes one peer transfer places on each measured shared tier."""
        timing = self.peer_transfer(payload_bytes, endpoints, dtype=dtype,
                                    topology=topology)
        return {f"tier:{name}": float(timing.payload_bytes)
                for name in timing.endpoint_path}

    def collective(self, kind: str, payload_bytes: float, ranks: int, *,
                   endpoints: Sequence[int | RankEndpoint] | None = None,
                   payloads_by_destination: Sequence[float] | None = None,
                   dtype: str | None = None, topology: str | None = None,
                   extrapolation_policy: str = "out_of_range",
                   **kwargs: Any) -> CollectiveTiming:
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        if kind not in COLLECTIVES or isinstance(ranks, bool) or ranks < 1:
            raise ModelingError(f"calibrated_collective_invalid:{kind}:{ranks}")
        if endpoints is not None:
            endpoints = _coerce_endpoints(list(endpoints))
            if len(endpoints) != int(ranks):
                raise ModelingError(
                    f"calibrated_endpoints_count:{len(endpoints)}!={ranks}")
        if kind == "mc2":
            one = self.collective(
                "all_to_all_v", payload, ranks, endpoints=endpoints,
                payloads_by_destination=payloads_by_destination, dtype=dtype,
                topology=topology, extrapolation_policy=extrapolation_policy,
                **kwargs)
            return replace(one, kind="mc2", total_us=2.0 * one.total_us,
                           bottleneck_us=2.0 * one.bottleneck_us,
                           wire_bytes_per_rank=2.0 * one.wire_bytes_per_rank,
                           tier_us=tuple((name, 2.0 * value) for name, value in one.tier_us),
                           rounds=tuple((name, 2 * count) for name, count in one.rounds),
                           effective_bw_gbps=one.effective_bw_gbps / 2.0,
                           method="calibrated_mc2_dispatch_combine")
        if ranks == 1:
            return CollectiveTiming(kind, ranks, payload, 0.0, (), "none", 0.0, 0.0,
                                    0.0, "trivial", (), (), True, "trivial")
        shells = [FabricTier(tier.name, tier.domain_ranks, 0.0, 1.0,
                             oversubscription=tier.oversubscription,
                             evidence="measured_profile") for tier in self.tiers]
        hops = _count_hops(shells, ranks)
        for shell, _groups, _before, _after in hops:
            mismatch = self._tier(shell.name).fit_mismatch(
                world_size=int(ranks), collective=str(kind),
                dtype=dtype, topology=topology)
            if mismatch is not None:
                raise ModelingError(
                    f"calibrated_fit_condition_mismatch:{shell.name}:{mismatch}")
        skew_rejected = False
        if payloads_by_destination is not None:
            values = tuple(float(value) for value in payloads_by_destination)
            if len(values) != int(ranks):
                raise ModelingError(
                    f"calibrated_all_to_all_destination_count:{len(values)}!={ranks}")
            uniform = len(set(values)) <= 1
            if not uniform and not self.skew_supported:
                # A uniform fitted curve has no evidence for a receiver
                # hotspot; fail closed instead of reporting the uniform number.
                skew_rejected = True
        flat_share = (ranks - 1) / float(ranks)
        flat_wire = (2.0 * payload * flat_share
                     if kind == "all_reduce" else payload * flat_share)
        per_tier, wire_total, rounds = [], 0.0, []
        unmeasured = False
        for shell, groups, before, after in hops:
            if payload == 0.0:
                first = self._tier(shell.name).points[0]
                measured = replace(first, payload_bytes=0.0, beta_ns_per_byte=0.0,
                                   congestion_factor=1.0, extrapolated=False)
            else:
                measured = self._tier(shell.name).at(
                    payload, policy=extrapolation_policy)
            if measured.load_basis == "unmeasured":
                unmeasured = True
            wire = flat_wire * max(0.0, (after - before) / float(max(1, ranks - 1)))
            count = (2 * (groups - 1) if kind == "all_reduce" else groups - 1)
            os_factor = (shell.oversubscription
                         if shell.semantics in {"shared", "mesh"} else 1.0)
            us = (measured.alpha_us * count + wire * measured.beta_ns_per_byte
                  * measured.congestion_factor * os_factor / 1000.0)
            per_tier.append((shell.name, us))
            rounds.append((shell.name, count))
            wire_total += wire
        total = float(sum(us for _, us in per_tier))
        bottleneck = max(per_tier, key=lambda item: item[1])
        qualified = not unmeasured and not skew_rejected
        evidence = ("calibrated_profile_uniform_only" if skew_rejected
                    else "unmeasured_profile" if unmeasured
                    else "measured_profile")
        return CollectiveTiming(kind, ranks, payload, total, tuple(per_tier),
                                bottleneck[0], float(bottleneck[1]), wire_total,
                                _effective_bw_gbps(payload, total),
                                "calibrated_piecewise_log2", tuple(rounds),
                                tuple(name for name, _ in per_tier),
                                qualified, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": "calibrated_profile", "source": self.source,
                "probe_seconds": self.probe_seconds,
                "assumptions": dict(self.assumptions),
                "skew_supported": bool(self.skew_supported),
                "tiers": [tier.to_dict() for tier in self.tiers]}


