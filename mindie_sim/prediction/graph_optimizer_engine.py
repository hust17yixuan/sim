"""Engine-backed analytical optimizer passes over a typed projection.

[Role] Discover UB-bounded Vector chains, Cube boundary regions, shared-pool
  prefetch and stream moves on the typed projection and evaluate them through
  the one public Engine.
[Invariants] Discovery is bounded before materializing tile lists; excess
  candidates are reported as ``unevaluated``; accepted plans are cumulative and
  preserve every declared physical work field.  Wide stream-region discovery is
  delegated to ``stream_discovery`` with an inner-work counter checked every 64
  edge/candidate ticks and a hard region cap.
[Timeouts & Locks] Bounded per-pass/total evaluation and wall budgets.
[Side-Effect Policy] Creates private Engine instances; no caller mutation.
"""
from __future__ import annotations

import math

from mindie_sim.prediction.engine_plan import (
    HardwareProfile, apply_fusion, apply_prefetch, apply_stream_move,
    audit_capture, baseline_plan, capture_counterfactual, effective_edges,
    evaluate_plan, necessary_sol, settle_prefetch, window_foreground_hbm,
)
from mindie_sim.prediction.fusion_graph import TILE_ROWS, TypedGraph, VECTOR_KINDS
from mindie_sim.prediction.fusion_feasibility import relaxed_requires_opt_in
from mindie_sim.prediction.fusion_recipe import (
    FusionCandidate, RejectedFusion, choose_vector_region, synthesize_cube_epilogue,
    synthesize_cube_prologue,
)
from mindie_sim.prediction.weight_prefetch import (
    PrefetchPhysics, plan_prefetch, residency_peak,
)

from .optimizer_report import Budget, OptimizationReport, PassReport, Rejection
from .stream_discovery import (
    adjacency as _adjacency, index_graph as _index,
    stream_regions as _stream_regions,
)


def _activation_inputs(graph: TypedGraph, op_id: str) -> int:
    return sum(1 for value_id in graph.ops[op_id].inputs
               if graph.values[value_id].role not in
               {"weight", "parameter", "scale", "state", "index"})


def _is_join(graph: TypedGraph, op_id: str) -> bool:
    op = graph.ops[op_id]
    return (op.resource == "vector" and op.kind in {"add", "elementwise", "mul", "bias"}
            and _activation_inputs(graph, op_id) > 1)


def _barrier(graph: TypedGraph, op_id: str) -> bool:
    op = graph.ops[op_id]
    return bool(op.collective or op.state_effect or op.capture_boundary
                or op.source_boundary or op.opaque)


def _vector_chains(graph: TypedGraph, limit: int = 6, *,
                   budget: Budget | None = None
                   ) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Bounded chain discovery over a pre-indexed adjacency.

    Returns ``(maximal_chains, aborted)``.  The wall/RSS budget is checked
    every 64 operators, before any chain is built, so a 2,000-operator graph
    with a 10 ms wall budget aborts inside discovery rather than after it.
    """
    chains: list[tuple[str, ...]] = []
    successors = _adjacency(graph, budget)
    if not successors and graph.ops:
        return (), True
    for index, (op_id, op) in enumerate(graph.ops.items()):
        if budget is not None and index and index % 64 == 0 and not budget.can_discover():
            return (), True
        if op.resource != "vector" or op.kind not in VECTOR_KINDS or _barrier(graph, op_id):
            continue
        chain = [op_id]
        current = op
        while len(chain) < limit:
            continuations: list[str] = []
            branch = False
            for successor in successors.get(current.op_id, ()):
                nxt = graph.ops[successor]
                if (nxt.resource == "vector" and nxt.kind in VECTOR_KINDS
                        and nxt.stage == op.stage and nxt.phase == op.phase
                        and not _barrier(graph, successor)
                        and _activation_inputs(graph, successor) <= 1):
                    continuations.append(successor)
                elif not _is_join(graph, successor):
                    branch = True
            if branch or len(continuations) != 1:
                break
            current = graph.ops[continuations[0]]
            chain.append(current.op_id)
        if len(chain) >= 2:
            chains.append(tuple(chain))
    maximal: list[tuple[str, ...]] = []
    for position, chain in enumerate(sorted(chains, key=lambda c: (-len(c), c))):
        if budget is not None and position and position % 64 == 0 \
                and not budget.can_discover():
            return (), True
        if any(prefix[:len(chain)] == chain for prefix in maximal
               if len(prefix) >= len(chain)):
            continue
        maximal.append(chain)
    return tuple(maximal), False


def _cube_epilogues(graph: TypedGraph, limit: int = 2, *,
                    adjacency: dict[str, tuple[str, ...]] | None = None,
                    consumers: dict[str, list[str]] | None = None,
                    budget: Budget | None = None
                    ) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], bool]:
    """Bounded Cube->Vector epilogue probes over the pre-indexed adjacency."""
    if adjacency is None or consumers is None:
        adjacency, consumers = _index(graph, budget)
        if adjacency is None or consumers is None:
            return (), True
    found: list[tuple[str, tuple[str, ...]]] = []
    for index, (op_id, op) in enumerate(graph.ops.items()):
        if budget is not None and index and index % 64 == 0 and not budget.can_discover():
            return tuple(found[:limit]), True
        if op.resource != "cube" or op.kind not in {"matmul", "grouped_matmul"}:
            continue
        for value_id in op.outputs:
            readers = consumers.get(value_id, ())
            if len(readers) != 1:
                continue
            first = readers[0]
            if graph.ops[first].resource != "vector" or graph.ops[first].kind not in {
                    "bias", "cast", "dynamic_quant", "quant", "swiglu", "add"}:
                continue
            found.append((op_id, (first,)))
            if len(found) >= limit:
                return tuple(found[:limit]), False
            second = adjacency.get(first, ())
            if len(second) == 1 and graph.ops[second[0]].resource == "vector" \
                    and graph.ops[second[0]].kind in {
                        "bias", "cast", "dynamic_quant", "quant", "swiglu"} \
                    and len(graph.ops[second[0]].inputs) == 1:
                found.append((op_id, (first, second[0])))
            if len(found) >= limit:
                return tuple(found[:limit]), False
    return tuple(found[:limit]), False


def _cube_prologues(graph: TypedGraph, *, budget: Budget | None = None
                    ) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Bounded Vector->Cube prologue probes over declared producer values."""
    found: list[tuple[str, str]] = []
    for index, (cube_id, cube) in enumerate(graph.ops.items()):
        if budget is not None and index and index % 64 == 0 and not budget.can_discover():
            return tuple(dict.fromkeys(found)), True
        if cube.resource != "cube":
            continue
        for value_id in cube.inputs:
            producer = graph.values[value_id].producer
            if producer is None or graph.ops.get(producer) is None:
                continue
            if graph.ops[producer].resource != "vector":
                continue
            found.append((producer, cube_id))
    return tuple(dict.fromkeys(found)), False




def _candidate_specs(pass_name: str, graph: TypedGraph, hardware: HardwareProfile,
                     budget: Budget, *, allow_relaxed_numerics: bool = False
                     ) -> tuple[tuple[FusionCandidate | RejectedFusion, ...], tuple[dict, ...]]:
    """Bounded discovery; every excess candidate is recorded as unevaluated."""
    found: list[FusionCandidate | RejectedFusion] = []
    unevaluated: list[dict] = []
    if not budget.can_discover():
        return (), ({"candidate_id": f"{pass_name}:discovery",
                      "pass": pass_name, "member_ops": [],
                      "reason": "discovery_budget_exhausted"},)
    if pass_name == "vector_fusion":
        chains, aborted = _vector_chains(graph, budget=budget)
        if aborted:
            return tuple(found), ({"candidate_id": "vector_chain:discovery",
                                   "pass": pass_name, "member_ops": [],
                                   "reason": "discovery_budget_exhausted"},)
        for chain in chains:
            for length in range(len(chain), 1, -1):
                members = chain[:length]
                for tile in reversed(TILE_ROWS):
                    if tile > graph.rows:
                        continue
                    candidate_id = f"vector_chain:{'+'.join(members)}:t{tile}"
                    if not budget.can_discover():
                        unevaluated.append({"candidate_id": candidate_id,
                                            "pass": pass_name,
                                            "member_ops": list(members),
                                            "reason": "discovery_budget_exhausted"})
                        return tuple(found), tuple(unevaluated)
                    if not budget.can_evaluate():
                        budget.mark(candidate_id, pass_name, members,
                                    "per_pass_candidate_budget_exhausted")
                        unevaluated.append({"candidate_id": candidate_id,
                                            "pass": pass_name,
                                            "member_ops": list(members),
                                            "reason": "per_pass_candidate_budget_exhausted"})
                        return tuple(found), tuple(unevaluated)
                    result = choose_vector_region(
                        graph, members, capacity_per_owner=hardware.ub_bytes_per_aiv,
                        tiles=(tile,), tensor_parallel=hardware.tensor_parallel,
                        usable_ub_bytes=hardware.usable_ub,
                        allow_relaxed_numerics=allow_relaxed_numerics)[0]
                    budget.charge()
                    found.append(result)
                    if isinstance(result, FusionCandidate):
                        break
    elif pass_name == "cube_epilogue":
        cube_ready = (hardware.cube_capacities is not None
                      and hardware.cube_capacities.complete)
        # An undeclared boundary capability needs one explicit direct probe,
        # not one charged discovery per tile for every candidate.
        epilogue_cap = hardware.capability("fixpipe_epilogue")
        prologue_cap = hardware.capability("cube_prologue")
        if cube_ready and (epilogue_cap or prologue_cap):
            adjacency, consumers = _index(graph, budget)
            if adjacency is None or consumers is None:
                unevaluated.append({
                    "candidate_id": f"{pass_name}:discovery",
                    "pass": pass_name, "member_ops": [],
                    "reason": "discovery_budget_exhausted"})
                return tuple(found), tuple(unevaluated)
            if epilogue_cap:
                epilogues, aborted = _cube_epilogues(
                    graph, adjacency=adjacency, consumers=consumers,
                    budget=budget)
                if aborted:
                    unevaluated.append({
                        "candidate_id": f"{pass_name}:discovery",
                        "pass": pass_name, "member_ops": [],
                        "reason": "discovery_budget_exhausted"})
                    return tuple(found), tuple(unevaluated)
                for cube, epilogue in epilogues:
                    for tile in (graph.rows, max(1, graph.rows // 2), 4, 2, 1):
                        if tile > graph.rows:
                            continue
                        candidate_id = (
                            f"cube_boundary:{'+'.join((cube, *epilogue))}:t{tile}")
                        if not budget.can_discover():
                            unevaluated.append({
                                "candidate_id": candidate_id,
                                "pass": pass_name,
                                "member_ops": [cube, *epilogue],
                                "reason": "discovery_budget_exhausted"})
                            return tuple(found), tuple(unevaluated)
                        if not budget.can_evaluate():
                            budget.mark(candidate_id, pass_name, (cube, *epilogue),
                                        "per_pass_candidate_budget_exhausted")
                            unevaluated.append({
                                "candidate_id": candidate_id,
                                "pass": pass_name,
                                "member_ops": [cube, *epilogue],
                                "reason": "per_pass_candidate_budget_exhausted"})
                            return tuple(found), tuple(unevaluated)
                        result = synthesize_cube_epilogue(
                            graph, cube, epilogue, tile_rows=tile,
                            capacity_per_owner=hardware.ub_bytes_per_aiv,
                            capability=hardware.capability("fixpipe_epilogue"),
                            cube_resources=hardware.cube_capacities,
                            tensor_parallel=hardware.tensor_parallel,
                            usable_ub_bytes=hardware.usable_ub,
                            allow_relaxed_numerics=allow_relaxed_numerics)
                        budget.charge()
                        found.append(result)
                        if isinstance(result, FusionCandidate):
                            break
            if prologue_cap:
                probes, aborted = _cube_prologues(graph, budget=budget)
                if aborted:
                    unevaluated.append({
                        "candidate_id": f"{pass_name}:discovery",
                        "pass": pass_name, "member_ops": [],
                        "reason": "discovery_budget_exhausted"})
                    return tuple(found), tuple(unevaluated)
                for producer, cube_id in probes:
                    for tile in (graph.rows, max(1, graph.rows // 2), 4, 2, 1):
                        if tile > graph.rows:
                            continue
                        candidate_id = (
                            f"cube_boundary:{producer}+{cube_id}:t{tile}")
                        if not budget.can_discover():
                            unevaluated.append({
                                "candidate_id": candidate_id,
                                "pass": pass_name,
                                "member_ops": [producer, cube_id],
                                "reason": "discovery_budget_exhausted"})
                            return tuple(found), tuple(unevaluated)
                        if not budget.can_evaluate():
                            budget.mark(candidate_id, pass_name,
                                        (producer, cube_id),
                                        "per_pass_candidate_budget_exhausted")
                            unevaluated.append({
                                "candidate_id": candidate_id,
                                "pass": pass_name,
                                "member_ops": [producer, cube_id],
                                "reason": "per_pass_candidate_budget_exhausted"})
                            return tuple(found), tuple(unevaluated)
                        result = synthesize_cube_prologue(
                            graph, producer, cube_id, tile_rows=tile,
                            capacity_per_owner=hardware.ub_bytes_per_aiv,
                            capability=hardware.capability("cube_prologue"),
                            retain_live_out=True,
                            cube_resources=hardware.cube_capacities,
                            tensor_parallel=hardware.tensor_parallel,
                            usable_ub_bytes=hardware.usable_ub,
                            allow_relaxed_numerics=allow_relaxed_numerics)
                        budget.charge()
                        found.append(result)
                        if isinstance(result, FusionCandidate):
                            break
    unique: dict[str, FusionCandidate | RejectedFusion] = {}
    for item in found:
        if isinstance(item, FusionCandidate) \
                and item.numerical_policy != "exact" \
                and not allow_relaxed_numerics \
                and relaxed_requires_opt_in(item.analysis.ops, item.member_ops):
            # Typed admission gate: nondeterministic rounding and relaxed
            # reductions are caller decisions, not optimizer defaults.
            item = RejectedFusion(
                item.candidate_id, item.pass_name, item.member_ops,
                ("relaxed_numerics_not_allowed",),
                "caller did not set allow_relaxed_numerics")
        unique.setdefault(item.candidate_id, item)
    return tuple(unique.values()), tuple(unevaluated)




def _run_analytical(graph: TypedGraph, hardware: HardwareProfile, *,
                    evaluation_budget: int, per_pass: int, max_streams: int | None,
                    wall_budget_s: float,
                    allow_relaxed_numerics: bool = False) -> OptimizationReport:
    graph.validate()
    budget = Budget(per_pass=per_pass, total=evaluation_budget, wall_s=wall_budget_s)
    plan = baseline_plan(graph, hardware)
    baseline = evaluate_plan(plan, hardware)
    budget.charge()
    current = baseline
    if baseline.status != "completed":
        return OptimizationReport(
            graph.rows, len(graph.ops), hardware, baseline, baseline, plan, plan, (),
            (Rejection("baseline", "baseline", tuple(graph.ops),
                       ("engine_evaluation_failed",), baseline.reason),),
            (), necessary_sol(plan, hardware),
            audit_capture(plan, hardware), budget.used_total, evaluation_budget,
            True, provenance="typed_projection_engine")
    passes: list[PassReport] = []
    rejections: list[Rejection] = []
    applied: list[dict] = []
    all_unevaluated: list[dict] = []

    def apply_pass(pass_name: str, stage: str) -> None:
        nonlocal plan, current
        budget.begin_pass()
        before = current
        pass_ids: list[str] = []
        pass_rejections: list[Rejection] = []
        while budget.can_evaluate() or budget.used_pass == 0:
            candidates, unevaluated = _candidate_specs(
                pass_name, graph, hardware, budget,
                allow_relaxed_numerics=allow_relaxed_numerics)
            all_unevaluated.extend(unevaluated)
            if not candidates:
                break
            best = None
            for result in candidates:
                if isinstance(result, RejectedFusion):
                    pass_rejections.append(Rejection(
                        result.candidate_id, result.pass_name, result.member_ops,
                        result.reasons, result.detail))
                    continue
                if not budget.can_evaluate():
                    budget.mark(result.candidate_id, pass_name, result.member_ops,
                                "per_pass_evaluation_budget_exhausted")
                    continue
                try:
                    trial = apply_fusion(plan, result)
                except ValueError as exc:
                    pass_rejections.append(Rejection(
                        result.candidate_id, pass_name, result.member_ops,
                        ("candidate_not_applicable",), str(exc)))
                    continue
                evaluated = evaluate_plan(trial, hardware)
                budget.charge()
                if evaluated.status != "completed":
                    pass_rejections.append(Rejection(
                        result.candidate_id, pass_name, result.member_ops,
                        ("engine_evaluation_failed",), evaluated.reason))
                    continue
                if best is None or evaluated.duration_ms < best[0]:
                    best = (evaluated.duration_ms, result, trial, evaluated)
            if best is None:
                break
            score, candidate, trial, evaluated = best
            threshold = max(1e-9, 1e-6 * current.duration_ms)
            if current.duration_ms - score <= threshold:
                pass_rejections.append(Rejection(
                    candidate.candidate_id, pass_name, candidate.member_ops,
                    ("no_improvement",),
                    f"engine {current.duration_ms:.9f} -> {score:.9f} ms"))
                break
            previous = current
            plan, current = trial, evaluated
            pass_ids.append(candidate.candidate_id)
            applied.append({"candidate_id": candidate.candidate_id, "pass": pass_name,
                            "members": list(candidate.member_ops),
                            "kernel": candidate.kernel_spec(),
                            "before_ms": previous.duration_ms, "after_ms": score,
                            "delta_ms": previous.duration_ms - score,
                            "hbm_before": previous.hbm_bytes,
                            "hbm_after": evaluated.hbm_bytes})
        passes.append(PassReport(stage, before.duration_ms, current.duration_ms,
                                 before.hbm_bytes, current.hbm_bytes, before.launches,
                                 current.launches, tuple(pass_ids), bool(pass_ids),
                                 tuple(r.candidate_id for r in pass_rejections),
                                 per_pass, budget.used_pass))
        rejections.extend(pass_rejections)

    apply_pass("vector_fusion", "+vector_fusion")
    apply_pass("cube_epilogue", "+cube_epilogue")
    # Pass 2.5: shared pool, explicit priority class, measured completion.
    budget.begin_pass()
    before = current
    pass_ids, pass_rejections = [], []
    residency: list[tuple[float, float, int]] = []
    for node in tuple(plan):
        if not budget.can_evaluate():
            break
        if node.kind != "kernel" or not node.work_dict().get("cube"):
            continue
        if any(p.prefetch_for == node.node_id for p in plan):
            continue
        launch_start = current.start(f"launch:{node.node_id}")
        window_start = (launch_start + hardware.launch_ms
                        if math.isfinite(launch_start) else 0.0)
        window_end = current.start(node.node_id)
        decision = plan_prefetch(
            graph, node.args[0] if node.args else node.node_id,
            window_ms=max(0.0, window_end - window_start),
            foreground_hbm_bytes=window_foreground_hbm(
                plan, current, window_start, window_end),
            physics=PrefetchPhysics(
                hardware.hbm_bytes_per_ms, hardware.onchip_bytes_per_ms,
                hardware.prefetch_capacity_bytes,
                tensor_parallel=hardware.tensor_parallel),
            capacity_bytes=hardware.prefetch_capacity_bytes,
            reserved_bytes=sum(amount for _, _, amount in residency),
            resident_bytes=0)
        if not decision.allowed:
            pass_rejections.append(Rejection(
                f"prefetch:{node.node_id}", "weight_prefetch", node.args,
                (decision.reason,), "capacity or sparse-window rejection"))
            continue
        if not budget.can_evaluate():
            budget.mark(f"prefetch:{node.node_id}", "weight_prefetch", node.args,
                        "per_pass_evaluation_budget_exhausted")
            break
        trial = apply_prefetch(plan, decision, node, hardware)
        measured = evaluate_plan(trial, hardware)
        budget.charge()
        prefetch_id = f"prefetch_W_{node.node_id}"
        completed = measured.prefetched(prefetch_id) if measured.status == "completed" else 0.0
        if completed <= 0:
            pass_rejections.append(Rejection(
                f"prefetch:{node.node_id}", "weight_prefetch", node.args,
                ("prefetch_did_not_complete_before_consumer",),
                f"completed={completed:.0f}B"))
            continue
        if not budget.can_evaluate():
            budget.mark(f"prefetch:{node.node_id}", "weight_prefetch", node.args,
                        "total_evaluation_budget_exhausted")
            break
        settled = settle_prefetch(trial, prefetch_id, completed)
        final_eval = evaluate_plan(settled, hardware)
        budget.charge()
        threshold = max(1e-9, 1e-6 * before.duration_ms)
        if final_eval.status != "completed" or \
                current.duration_ms - final_eval.duration_ms <= threshold:
            pass_rejections.append(Rejection(
                f"prefetch:{node.node_id}", "weight_prefetch", node.args,
                ("no_improvement",),
                f"engine {current.duration_ms:.9f} -> {final_eval.duration_ms:.9f} ms"))
            continue
        residency.append((measured.start(prefetch_id), measured.complete(node.node_id),
                          int(completed)))
        plan, previous, current = settled, current, final_eval
        pass_ids.append(f"prefetch:{node.node_id}:{int(completed)}")
        applied.append({"candidate_id": f"prefetch:{node.node_id}",
                        "pass": "weight_prefetch", "members": list(node.args),
                        "decision": decision.settled(int(completed)).record(),
                        "before_ms": previous.duration_ms,
                        "after_ms": final_eval.duration_ms,
                        "delta_ms": previous.duration_ms - final_eval.duration_ms,
                        "hbm_before": previous.hbm_bytes,
                        "hbm_after": final_eval.hbm_bytes})
    peak, _ = residency_peak(tuple(residency))
    passes.append(PassReport("+weight_prefetch", before.duration_ms, current.duration_ms,
                             before.hbm_bytes, current.hbm_bytes, before.launches,
                             current.launches, tuple(pass_ids), bool(pass_ids),
                             tuple(r.candidate_id for r in pass_rejections),
                             per_pass, budget.used_pass))
    rejections.extend(pass_rejections)
    # Pass 3: streams, bounded by the declared hardware count (no caller bypass).
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
    was_exhausted = budget.exhausted
    stream_regions = _stream_regions(plan, streams, budget)
    if budget.exhausted and (not was_exhausted or not stream_regions):
        all_unevaluated.append({
            "candidate_id": "stream:discovery", "pass": "streams",
            "member_ops": [], "reason": "discovery_budget_exhausted"})
    for region in stream_regions[:per_pass]:
        if not budget.can_evaluate():
            budget.mark("stream:" + "+".join(region), "streams", region,
                        "per_pass_evaluation_budget_exhausted")
            break
        try:
            trial = apply_stream_move(plan, region, "aux", hardware=hardware)
        except ValueError as exc:
            pass_rejections.append(Rejection("stream:" + "+".join(region), "streams",
                                             region, ("stream_move_rejected",), str(exc)))
            continue
        evaluated = evaluate_plan(trial, hardware)
        budget.charge()
        threshold = max(1e-9, 1e-6 * current.duration_ms)
        if evaluated.status == "completed" and \
                current.duration_ms - evaluated.duration_ms > threshold:
            previous = current
            plan, current = trial, evaluated
            pass_ids.append("stream:" + "+".join(region))
            applied.append({"candidate_id": "stream:" + "+".join(region),
                            "pass": "streams", "members": list(region),
                            "before_ms": previous.duration_ms,
                            "after_ms": evaluated.duration_ms,
                            "delta_ms": previous.duration_ms - evaluated.duration_ms,
                            "hbm_before": previous.hbm_bytes,
                            "hbm_after": evaluated.hbm_bytes})
        else:
            pass_rejections.append(Rejection(
                "stream:" + "+".join(region), "streams", region,
                ("no_improvement",), "engine stream trial"))
    passes.append(PassReport("+multistream_overlap", before.duration_ms,
                             current.duration_ms, before.hbm_bytes, current.hbm_bytes,
                             before.launches, current.launches, tuple(pass_ids),
                             bool(pass_ids),
                             tuple(r.candidate_id for r in pass_rejections),
                             per_pass, budget.used_pass))
    rejections.extend(pass_rejections)
    if budget.can_evaluate():
        _captured, capture = capture_counterfactual(plan, hardware, current)
        budget.charge()
    else:
        _captured, capture = None, audit_capture(plan, hardware)
    return OptimizationReport(
        graph.rows, len(graph.ops), hardware, baseline, current,
        baseline_plan(graph, hardware), plan, tuple(passes), tuple(rejections),
        tuple(applied), necessary_sol(plan, hardware), capture, budget.used_total,
        evaluation_budget, budget.exhausted, tuple(all_unevaluated), per_pass,
        "typed_projection_engine", None, effective_edges(plan))


