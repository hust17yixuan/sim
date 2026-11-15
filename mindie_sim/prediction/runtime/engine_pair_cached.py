"""Cached two-task steady-state rate plans and single-step integration.

[Role] Memoize static two-task arbitration plans keyed by ``_PreparedTask``
  identities and integrate cached steps without rebuilding rate dictionaries.
[Invariants] Zero allocated service rates raise ``service_rate_underflow``
  before any state mutation; 4-ULP snap matches ``service_step`` bit-for-bit.
[Timeouts & Locks] Bounded 4,096-entry cache; no locks, sleeps or I/O.
[Side-Effect Policy] Mutates live task remaining/service maps and Engine counters.
"""
from __future__ import annotations

import heapq
import math

from .engine_types import ExecutionError, _PreparedTask, _Task


class _EnginePairCachedMixin:
    def _cache_pair_plan(self, p0: _PreparedTask, p1: _PreparedTask,
                         t0: _Task, cp_fast) -> None:
        pair_cache = self._pair_rate_cache
        if len(pair_cache) >= 4096:
            pair_cache.clear()
        pkey = (id(p0), id(p1))
        if cp_fast is None:
            pair_cache[pkey] = (p0, p1, None)
            return
        m = cp_fast[0]
        if m == 3:
            _, _, tc, c_rates, _, tu, name_u, rate_u = cp_fast
            pc = tc.prepared
            pu = tu.prepared
            if (pc is not None and pc.coupled_demand is not None
                    and pu is not None and pu.uncoupled_res_info is not None
                    and name_u in pu.uncoupled_res_info):
                init_r_u = pu.uncoupled_res_info[name_u][0]
                if tu.remaining[name_u] == init_r_u:
                    pair_cache[pkey] = (
                        p0, p1,
                        (3, tc is t0, c_rates, name_u, rate_u, True, init_r_u),
                    )
                return
        elif m == 5:
            _, _, tc, c_items_t, fc, _, tu, name_u, rate_u = cp_fast
            pc = tc.prepared
            pu = tu.prepared
            if (pc is not None and pc.coupled_demand is not None
                    and pu is not None and pu.uncoupled_res_info is not None
                    and name_u in pu.uncoupled_res_info):
                init_r_u = pu.uncoupled_res_info[name_u][0]
                if tu.remaining[name_u] == init_r_u:
                    c_rates = (tuple((rn, amt * fc) for rn, amt in c_items_t)
                               if fc else ())
                    pair_cache[pkey] = (
                        p0, p1,
                        (3, tc is t0, c_rates, name_u, rate_u, not fc, init_r_u),
                    )
                return
        elif m == 4:
            _, _, _, n0, rate0, _, _, n1, rate1 = cp_fast
            pair_cache[pkey] = (p0, p1, (4, n0, rate0, n1, rate1))
            return
        elif m == 1:
            if p0.coupled_demand is not None and p1.coupled_demand is not None:
                _, _, _, dem0, f0, _, _, dem1, f1 = cp_fast
                rates0 = (tuple((rn, amt * f0) for rn, amt in dem0.items())
                          if f0 else ())
                rates1 = (tuple((rn, amt * f1, (not f0 or rn not in dem0))
                                for rn, amt in dem1.items())
                          if f1 else ())
                pair_cache[pkey] = (p0, p1, (1, rates0, rates1))
                return
        pair_cache[pkey] = (p0, p1, False)

    def _step_cached_pair(
        self, cplan: tuple, k0: int, t0: _Task, k1: int, t1: _Task,
        now: float, dt: float,
    ) -> float:
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        cm = cplan[0]
        if cm == 3:
            _, tc_is_0, c_rates, name_u, rate_u, u_busy, _ = cplan
            if tc_is_0:
                kc, tc, ku, tu = k0, t0, k1, t1
            else:
                kc, tc, ku, tu = k1, t1, k0, t0
            rem_c = tc.remaining
            rem_u = tu.remaining
            for rname, rate_c in c_rates:
                if rate_c == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = rem_c[rname] / rate_c
                if wdt < dt:
                    dt = wdt
            if rate_u:
                wdt_u = rem_u[name_u] / rate_u
                if wdt_u < dt:
                    dt = wdt_u
            if not _isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = _nextafter(now, _inf)
            if c_rates:
                svc_c = tc.service
                for rname, rate_c in c_rates:
                    busy_ms[rname] += dt
                    remaining = rem_c[rname]
                    scaled = dt * rate_c
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * _ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc_c:
                        svc_c[rname] += used
                    else:
                        svc_c[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem_c[rname] = residual
                    else:
                        del rem_c[rname]
                if not rem_c and tc.latency <= 0.0:
                    completable_seen.add(kc)
                    _heappush(completable, kc)
            if rate_u:
                svc_u = tu.service
                if u_busy:
                    busy_ms[name_u] += dt
                remaining = rem_u[name_u]
                scaled = dt * rate_u
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * _ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                if name_u in svc_u:
                    svc_u[name_u] += used
                else:
                    svc_u[name_u] = used
                serviced[name_u] += used
                if residual:
                    rem_u[name_u] = residual
                else:
                    del rem_u[name_u]
                    if tu.latency <= 0.0:
                        completable_seen.add(ku)
                        _heappush(completable, ku)
            return next_time
        if cm == 4:
            _, n0, rate0, n1, rate1 = cplan
            rem0 = t0.remaining
            rem1 = t1.remaining
            if rate0 == 0.0:
                raise ExecutionError(f"service_rate_underflow:{n0}")
            wdt0 = rem0[n0] / rate0
            if wdt0 < dt:
                dt = wdt0
            if rate1 == 0.0:
                raise ExecutionError(f"service_rate_underflow:{n1}")
            wdt1 = rem1[n1] / rate1
            if wdt1 < dt:
                dt = wdt1
            if not _isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = _nextafter(now, _inf)
            busy_ms[n0] += dt
            rem_v0 = rem0[n0]
            scaled0 = dt * rate0
            if scaled0 >= rem_v0:
                used0, res0 = rem_v0, 0.0
            else:
                res0 = rem_v0 - scaled0
                if res0 <= 4 * _ulp(rem_v0):
                    used0, res0 = rem_v0, 0.0
                else:
                    used0 = scaled0
            svc0 = t0.service
            if n0 in svc0:
                svc0[n0] += used0
            else:
                svc0[n0] = used0
            serviced[n0] += used0
            if res0:
                rem0[n0] = res0
            else:
                del rem0[n0]
                if t0.latency <= 0.0:
                    completable_seen.add(k0)
                    _heappush(completable, k0)
            busy_ms[n1] += dt
            rem_v1 = rem1[n1]
            scaled1 = dt * rate1
            if scaled1 >= rem_v1:
                used1, res1 = rem_v1, 0.0
            else:
                res1 = rem_v1 - scaled1
                if res1 <= 4 * _ulp(rem_v1):
                    used1, res1 = rem_v1, 0.0
                else:
                    used1 = scaled1
            svc1 = t1.service
            if n1 in svc1:
                svc1[n1] += used1
            else:
                svc1[n1] = used1
            serviced[n1] += used1
            if res1:
                rem1[n1] = res1
            else:
                del rem1[n1]
                if t1.latency <= 0.0:
                    completable_seen.add(k1)
                    _heappush(completable, k1)
            return next_time
        _, rates0, rates1 = cplan
        rem0 = t0.remaining
        rem1 = t1.remaining
        for rname, rate0 in rates0:
            if rate0 == 0.0:
                raise ExecutionError(f"service_rate_underflow:{rname}")
            wdt = rem0[rname] / rate0
            if wdt < dt:
                dt = wdt
        for rname, rate1, _ in rates1:
            if rate1 == 0.0:
                raise ExecutionError(f"service_rate_underflow:{rname}")
            wdt = rem1[rname] / rate1
            if wdt < dt:
                dt = wdt
        if not _isfinite(dt):
            names = ",".join(t.event.name for t in self._tasks.values())
            raise ExecutionError(f"execution_stalled:{names[:300]}")
        next_time = now + dt
        if dt > 0 and next_time == now:
            next_time = _nextafter(now, _inf)
        if rates0:
            svc0 = t0.service
            for rname, rate0 in rates0:
                busy_ms[rname] += dt
                remaining = rem0[rname]
                scaled = dt * rate0
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * _ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                if rname in svc0:
                    svc0[rname] += used
                else:
                    svc0[rname] = used
                serviced[rname] += used
                if residual:
                    rem0[rname] = residual
                else:
                    del rem0[rname]
            if not rem0 and t0.latency <= 0.0:
                completable_seen.add(k0)
                _heappush(completable, k0)
        if rates1:
            svc1 = t1.service
            for rname, rate1, add_busy in rates1:
                if add_busy:
                    busy_ms[rname] += dt
                remaining = rem1[rname]
                scaled = dt * rate1
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * _ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                if rname in svc1:
                    svc1[rname] += used
                else:
                    svc1[rname] = used
                serviced[rname] += used
                if residual:
                    rem1[rname] = residual
                else:
                    del rem1[rname]
            if not rem1 and t1.latency <= 0.0:
                completable_seen.add(k1)
                _heappush(completable, k1)
        return next_time
