"""Canonical Case/ModelSpec optimizer passes scored by ``graph_costs``.

[Role] Rewrite the selected OpSpec tuple for each candidate and evaluate it
  through the common graph-cost/Execution path.
[Invariants] Only the proven eliminated HBM access changes; every other
  declared service field survives. Prefetch credit is measured completion before
  the consumer start under a strictly lower priority class.
[Timeouts & Locks] Bounded per-pass/total evaluation and wall budgets.
[Side-Effect Policy] Returns new immutable CanonicalPlan records.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.capture_barriers import standalone_capture_blockers
from mindie_sim.prediction.engine_plan import necessary_sol
from mindie_sim.prediction.fusion_feasibility import relaxed_requires_opt_in
from mindie_sim.prediction.fusion_recipe import RejectedFusion
from mindie_sim.prediction.fusion_bridge import (
    CanonicalPlan, add_prefetch, apply_prefetch_rewrite, canonical_stream_move,
    evaluate_rewrite, prefetch_completed_before, project_graph, rewrite_fusion,
    settle_prefetch_rewrite, timeline_by_op,
)
from mindie_sim.prediction.fusion_pool import (
    PoolLedger, declared_l1_holders, l1_peak_holders,
)
from mindie_sim.prediction.weight_prefetch import (
    PrefetchPhysics, plan_prefetch, static_weight_values,
)

from .graph_optimizer_engine import _candidate_specs
from .optimizer_report import _canonical_dag, _canonical_eval, _canonical_nodes
from .optimizer_report import Budget, OptimizationReport, PassReport, Rejection


def _run_canonical(plan: CanonicalPlan, *, evaluation_budget: int, per_pass: int,
                   max_streams: int | None, wall_budget_s: float,
                   allow_relaxed_numerics: bool = False) -> OptimizationReport:
    hardware = plan.hardware
    plan.graph.validate()
    budget = Budget(per_pass=per_pass, total=evaluation_budget, wall_s=wall_budget_s)
    try:
        baseline_result = plan.evaluate()
    finally:
        # A failed canonical evaluation still consumed a real attempt.
        budget.charge()
    baseline_nodes = _canonical_nodes(plan)
    baseline = _canonical_eval(baseline_result, baseline_nodes)
    passes: list[PassReport] = []
    rejections: list[Rejection] = []
    applied: list[dict] = []
    all_unevaluated: list[dict] = []
    current_plan = plan
    current = baseline
    current_result = baseline_result

    def accept(result, candidate, previous, pass_name, members):
        nonlocal current_plan, current, current_result
        current_plan = result[0]
        current_result = result[1]
        current = _canonical_eval(current_result, _canonical_nodes(current_plan))
        applied.append({"candidate_id": candidate.candidate_id, "pass": pass_name,
                        "members": list(members), "kernel": candidate.kernel_spec(),
                        "before_ms": previous.duration_ms,
                        "after_ms": current.duration_ms,
                        "delta_ms": previous.duration_ms - current.duration_ms,
                        "hbm_before": previous.hbm_bytes,
                        "hbm_after": current.hbm_bytes})

    for pass_name, stage in (("vector_fusion", "+vector_fusion"),
                             ("cube_epilogue", "+cube_epilogue")):
        budget.begin_pass()
        before = current
        pass_ids: list[str] = []
        pass_rejections: list[Rejection] = []
        while budget.can_evaluate() or budget.used_pass == 0:
            specs, unevaluated = _candidate_specs(
                pass_name, current_plan.graph, hardware, budget,
                allow_relaxed_numerics=allow_relaxed_numerics)
            all_unevaluated.extend(unevaluated)
            if not specs:
                break
            best = None
            for spec in specs:
                if isinstance(spec, RejectedFusion):
                    pass_rejections.append(Rejection(spec.candidate_id, spec.pass_name,
                                                     spec.member_ops, spec.reasons,
                                                     spec.detail))
                    continue
                if spec.numerical_policy != "exact" and not allow_relaxed_numerics \
                        and relaxed_requires_opt_in(spec.analysis.ops,
                                                    spec.member_ops):
                    # Canonical admission gate: nondeterministic rounding and
                    # relaxed reductions are caller decisions at the public
                    # entry point, independent of reduction membership.
                    pass_rejections.append(Rejection(
                        spec.candidate_id, spec.pass_name, spec.member_ops,
                        ("relaxed_numerics_not_allowed",),
                        "caller did not set allow_relaxed_numerics"))
                    continue
                if not budget.can_evaluate():
                    budget.mark(spec.candidate_id, pass_name, spec.member_ops,
                                "per_pass_evaluation_budget_exhausted")
                    continue
                try:
                    rewrite = rewrite_fusion(current_plan, spec)
                    updated, result = evaluate_rewrite(current_plan, rewrite)
                except (ValueError, KeyError) as exc:
                    pass_rejections.append(Rejection(
                        spec.candidate_id, pass_name, spec.member_ops,
                        ("canonical_rewrite_rejected",), str(exc)))
                    continue
                finally:
                    budget.charge()
                evaluated = _canonical_eval(result, _canonical_nodes(updated))
                if best is None or evaluated.duration_ms < best[0]:
                    best = (evaluated.duration_ms, spec, (updated, result))
            if best is None:
                break
            score, candidate, result = best
            threshold = max(1e-9, 1e-6 * current.duration_ms)
            if current.duration_ms - score <= threshold:
                pass_rejections.append(Rejection(
                    candidate.candidate_id, pass_name, candidate.member_ops,
                    ("no_improvement",),
                    f"graph_costs {current.duration_ms:.9f} -> {score:.9f} ms"))
                break
            previous = current
            accept(result, candidate, previous, pass_name, candidate.member_ops)
            pass_ids.append(candidate.candidate_id)
        passes.append(PassReport(stage, before.duration_ms, current.duration_ms,
                                 before.hbm_bytes, current.hbm_bytes, before.launches,
                                 current.launches, tuple(pass_ids), bool(pass_ids),
                                 tuple(r.candidate_id for r in pass_rejections),
                                 per_pass, budget.used_pass))
        rejections.extend(pass_rejections)

    # Pass 2.5: canonical prefetch with a measured completion window.
    budget.begin_pass()
    before = current
    pass_ids, pass_rejections = [], []
    residency = PoolLedger()
    audit_result = None
    trace = ()
    if budget.can_evaluate():
        try:
            audit_result = current_plan.evaluate(audit_allocations=True)
        finally:
            budget.charge()
        trace = tuple(audit_result.get("allocation_trace", ()))
    already_prefetched = {str(dict(other.metadata).get("prefetch_target", ""))
                          for other in current_plan.ops
                          if other.family == "prefetch"}
    for op in (tuple(current_plan.ops) if audit_result is not None else ()):
        if not budget.can_discover():
            break
        if op.op_id.startswith("prefetch_W_") or op.op_id.startswith("release_W_") \
                or op.family in {"prefetch", "release", "event", "native_fusion"}:
            continue
        if op.op_id in already_prefetched:
            continue
        if not static_weight_values(current_plan.graph, op.op_id):
            continue
        row = timeline_by_op(audit_result).get(op.op_id)
        if row is None:
            continue
        t_c = float(row["ready_ms"])
        foreground = sum(other.hbm_bytes for other in current_plan.ops
                         if other.op_id in op.deps)
        holders = max(declared_l1_holders(current_plan),
                      l1_peak_holders(trace, 0.0, t_c))
        decision = plan_prefetch(
            current_plan.graph, op.op_id, window_ms=t_c,
            foreground_hbm_bytes=foreground,
            physics=PrefetchPhysics(
                hardware.hbm_bytes_per_ms, hardware.onchip_bytes_per_ms,
                hardware.prefetch_capacity_bytes,
                tensor_parallel=hardware.tensor_parallel),
            capacity_bytes=hardware.prefetch_capacity_bytes,
            tensor_parallel=hardware.tensor_parallel,
            reserved_bytes=holders + residency.reserved)
        if not decision.allowed:
            pass_rejections.append(Rejection(
                f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                (decision.reason,), "capacity/window rejection"))
            continue
        target = decision.target_bytes
        for _attempt in range(4):
            if not budget.can_evaluate():
                budget.mark(f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                            "per_pass_evaluation_budget_exhausted")
                break
            added = add_prefetch(current_plan, decision, hardware)
            if added is None:
                break
            trial_plan, prefetch_id = added
            try:
                trial_result = trial_plan.evaluate()
            except TimeoutError:
                raise
            except Exception as exc:
                pass_rejections.append(Rejection(
                    f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                    ("prefetch_pool_blocked",), f"{type(exc).__name__}:{exc}"[:240]))
                break
            finally:
                budget.charge()
            trial_rows = timeline_by_op(trial_result)
            marker, complete, ready = prefetch_completed_before(
                trial_result, prefetch_id, op.op_id)
            done = target if marker and complete <= ready + 1e-12 else 0
            if done <= 0:
                target //= 2
                if target <= 0:
                    break
                decision = type(decision)(**{**decision.__dict__, "target_bytes": target,
                                             "bytes_prefetched": target,
                                             "hbm_read_bytes": decision.weight_bytes - target,
                                             "onchip_read_bytes": target})
                continue
            if not budget.can_evaluate():
                budget.mark(f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                            "total_evaluation_budget_exhausted")
                break
            rewrite = settle_prefetch_rewrite(trial_plan, decision, prefetch_id, done,
                                              hardware)
            if rewrite is None:
                break
            settled_plan = apply_prefetch_rewrite(trial_plan, rewrite, prefetch_id)
            try:
                settled_result = settled_plan.evaluate()
            except TimeoutError:
                raise
            except Exception as exc:
                pass_rejections.append(Rejection(
                    f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                    ("settled_prefetch_pool_blocked",),
                    f"{type(exc).__name__}:{exc}"[:240]))
                break
            finally:
                budget.charge()
            settled = _canonical_eval(settled_result, _canonical_nodes(settled_plan))
            threshold = max(1e-9, 1e-6 * before.duration_ms)
            if current.duration_ms - settled.duration_ms <= threshold:
                pass_rejections.append(Rejection(
                    f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                    ("no_improvement",),
                    f"graph_costs {current.duration_ms:.9f} -> "
                    f"{settled.duration_ms:.9f} ms"))
                break
            prefetch_row = trial_rows.get(prefetch_id, {})
            consumer_row = trial_rows.get(op.op_id, {})
            prefetch_start = float(prefetch_row.get("ready_ms", 0.0))
            consumer_end = float(consumer_row.get("complete_ms", ready))
            # Residency starts at copy start and ends when the consumer
            # retires, not at copy completion.
            interval_count = len(residency.intervals)
            residency.add(prefetch_start, consumer_end, int(done))
            if len(residency.intervals) > interval_count \
                    and residency.peak() > hardware.prefetch_capacity_bytes:
                residency.intervals.pop()
                pass_rejections.append(Rejection(
                    f"prefetch:{op.op_id}", "weight_prefetch", (op.op_id,),
                    ("prefetch_pool_overbooked",),
                    f"peak {residency.peak()}B > C={hardware.prefetch_capacity_bytes}B"))
                break
            previous = current
            current_plan, current_result = settled_plan, settled_result
            current = settled
            pass_ids.append(f"prefetch:{op.op_id}:{int(done)}")
            applied.append({"candidate_id": f"prefetch:{op.op_id}",
                            "pass": "weight_prefetch", "members": [op.op_id],
                            "decision": decision.settled(int(done)).record(),
                            "pool_holders_bytes": holders,
                            "residency_ms": [prefetch_start, consumer_end],
                            "before_ms": previous.duration_ms,
                            "after_ms": current.duration_ms,
                            "delta_ms": previous.duration_ms - current.duration_ms,
                            "hbm_before": previous.hbm_bytes,
                            "hbm_after": current.hbm_bytes})
            break
    passes.append(PassReport("+weight_prefetch", before.duration_ms, current.duration_ms,
                             before.hbm_bytes, current.hbm_bytes, before.launches,
                             current.launches, tuple(pass_ids), bool(pass_ids),
                             tuple(r.candidate_id for r in pass_rejections),
                             per_pass, budget.used_pass))
    rejections.extend(pass_rejections)
    # Pass 3: canonical stream moves.
    declared_streams = hardware.max_streams
    requested = int(max_streams or declared_streams)
    streams = min(requested, declared_streams)
    budget.begin_pass()
    before = current
    pass_ids, pass_rejections = [], []
    if requested > declared_streams:
        pass_rejections.append(Rejection("streams:declared_cap", "streams", (),
                                         ("requested_streams_exceed_declared",),
                                         f"{requested} > {declared_streams}"))
    regions = _canonical_stream_regions(current_plan, streams, budget)
    for region in regions[:per_pass]:
        if not budget.can_evaluate():
            budget.mark("stream:" + "+".join(region), "streams", region,
                        "per_pass_evaluation_budget_exhausted")
            break
        try:
            ops = canonical_stream_move(current_plan, region, hardware, "aux")
            trial_plan = replace(current_plan, ops=ops,
                                 graph=project_graph(current_plan.case, ops))
            trial_result = trial_plan.evaluate()
        except ValueError as exc:
            pass_rejections.append(Rejection("stream:" + "+".join(region), "streams",
                                             region, ("stream_move_rejected",), str(exc)))
            continue
        finally:
            budget.charge()
        evaluated = _canonical_eval(trial_result, _canonical_nodes(trial_plan))
        threshold = max(1e-9, 1e-6 * current.duration_ms)
        if current.duration_ms - evaluated.duration_ms > threshold:
            previous = current
            current_plan, current_result, current = trial_plan, trial_result, evaluated
            pass_ids.append("stream:" + "+".join(region))
            applied.append({"candidate_id": "stream:" + "+".join(region),
                            "pass": "streams", "members": list(region),
                            "before_ms": previous.duration_ms,
                            "after_ms": current.duration_ms,
                            "delta_ms": previous.duration_ms - current.duration_ms,
                            "hbm_before": previous.hbm_bytes,
                            "hbm_after": current.hbm_bytes})
        else:
            pass_rejections.append(Rejection(
                "stream:" + "+".join(region), "streams", region,
                ("no_improvement",), "canonical stream trial"))
    passes.append(PassReport("+multistream_overlap", before.duration_ms,
                             current.duration_ms, before.hbm_bytes, current.hbm_bytes,
                             before.launches, current.launches, tuple(pass_ids),
                             bool(pass_ids),
                             tuple(r.candidate_id for r in pass_rejections),
                             per_pass, budget.used_pass))
    rejections.extend(pass_rejections)
    # Capture: evaluate a real replay counterfactual, never an arithmetic guess.
    blockers = []
    for op in current_plan.ops:
        meta = dict(op.metadata)
        node = current_plan.graph.ops.get(op.op_id)
        blockers.extend(standalone_capture_blockers(op, node))
        # A prefetch/release lease has no stable pointer unless the region
        # explicitly declared one and the hardware exposes graph prefetch.
        if op.family in {"prefetch", "release"} and (
                meta.get("pointer_stable") is not True
                or not hardware.capability("aclgraph_prefetch")):
            blockers.append(f"prefetch_pointer_stability_unknown:{op.op_id}")
    capture_evaluated = None
    capture_saved = None
    if (not blockers and hardware.capability("aclgraph_capture")
            and budget.can_evaluate()):
        try:
            capture_result = current_plan.evaluate(capture_mode="full")
        finally:
            budget.charge()
        capture_evaluated = _canonical_eval(
            capture_result, _canonical_nodes(current_plan))
        capture_saved = max(0.0, current.duration_ms - capture_evaluated.duration_ms)
    capture = {"status": "blocked" if blockers else (
        "eligible" if capture_evaluated is not None else "unknown"),
        "mode": "aclgraph_full", "blockers": tuple(dict.fromkeys(blockers)),
        "host_bubble_saved_ms": capture_saved,
        "evaluated": capture_evaluated is not None,
        "normal_ms": current.duration_ms if capture_evaluated is not None else None,
        "capture_ms": capture_evaluated.duration_ms if capture_evaluated else None}
    final_nodes = _canonical_nodes(current_plan)
    return OptimizationReport(
        current_plan.graph.rows, len(current_plan.ops), hardware, baseline, current,
        baseline_nodes, final_nodes, tuple(passes), tuple(rejections), tuple(applied),
        necessary_sol(final_nodes, hardware,
                      current_plan.external_inputs), capture, budget.used_total,
        evaluation_budget, budget.exhausted, tuple(all_unevaluated), per_pass,
        "canonical_graph_costs", current_plan,
        _canonical_dag(current_plan, current_result))


def _canonical_stream_regions(plan: CanonicalPlan, streams: int,
                              budget: Budget | None = None
                              ) -> tuple[tuple[str, ...], ...]:
    """Bounded stream regions over one O(V + E) successor index.

    ``graph.successors`` is a full operator scan, so the previous nested walk
    was quadratic in the candidate graph.  The index is built once and the
    caller's wall/RSS budget is checked every 64 operators.
    """
    if streams < 2:
        return ()
    if budget is not None and not budget.can_discover():
        return ()
    by_id = {op.op_id: op for op in plan.ops}
    graph = plan.graph
    consumers: dict[str, list[str]] = {}
    for index, node in enumerate(graph.ops.values()):
        if budget is not None and index % 64 == 0 and not budget.can_discover():
            return ()
        for value_id in node.inputs:
            consumers.setdefault(value_id, []).append(node.op_id)
    dependency: dict[str, list[str]] = {op_id: [] for op_id in graph.ops}
    for index, node in enumerate(graph.ops.values()):
        if budget is not None and index % 64 == 0 and not budget.can_discover():
            return ()
        for dep in node.deps:
            if dep in dependency and dep != node.op_id:
                dependency[dep].append(node.op_id)
    successors: dict[str, tuple[str, ...]] = {}
    for index, (op_id, node) in enumerate(graph.ops.items()):
        if budget is not None and index % 64 == 0 and not budget.can_discover():
            return ()
        found: list[str] = []
        seen: set[str] = set()
        for value_id in node.outputs:
            for consumer in consumers.get(value_id, ()):
                if consumer != op_id and consumer not in seen:
                    seen.add(consumer)
                    found.append(consumer)
        for successor in dependency.get(op_id, ()):
            if successor not in seen:
                seen.add(successor)
                found.append(successor)
        successors[op_id] = tuple(found)
    regions = []
    checks = 0

    def movable(op_id: str) -> bool:
        op = by_id.get(op_id)
        node = graph.ops.get(op_id)
        return (op is not None and node is not None and (op.stream or "main") == "main"
                and not node.collective and not node.state_effect
                and not node.capture_boundary and node.resource != "host"
                and op.family not in {"prefetch", "native_fusion"})

    for fork in plan.ops:
        checks += 1
        if budget is not None and checks % 64 == 0 and not budget.can_discover():
            break
        branches = [s for s in successors.get(fork.op_id, ()) if movable(s)]
        if len(branches) < 2:
            continue
        for branch in branches:
            region = [branch]
            cursor = branch
            while True:
                checks += 1
                if budget is not None and checks % 64 == 0 \
                        and not budget.can_discover():
                    return tuple(dict.fromkeys(regions))
                nxt = [s for s in successors.get(cursor, ()) if movable(s)]
                if len(nxt) != 1:
                    break
                cursor = nxt[0]
                region.append(cursor)
            if 2 <= len(region) <= 8 and all(movable(item) for item in region):
                regions.append(tuple(region))
    return tuple(dict.fromkeys(regions))

