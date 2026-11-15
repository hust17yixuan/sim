"""Typed optimizer budget and report records shared by both execution paths.

[Role] Hold the per-pass/total evaluation budget, the monotone pass records and
  the exported DAG/report projection.
[Invariants] ``Budget`` shares one candidate/Engine-evaluation counter across a
  whole pass. The exported DAG is the effective execution projection, including
  prefetch joins, stream order and reservation releases.
[Timeouts & Locks] Monotone wall/RSS checks; no threads or locks.
[Side-Effect Policy] Pure records over caller-supplied plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
try:
    import resource
except ImportError:  # pragma: no cover - Windows has no POSIX resource module
    resource = None  # type: ignore[assignment]
import sys
import time

from mindie_sim.prediction.engine_plan import (
    EngineEvaluation, HardwareProfile, PlanNode, effective_edges,
)
from mindie_sim.prediction.fusion_bridge import CanonicalPlan, timeline_by_op

MAX_PEAK_BYTES = 512 * 1024 * 1024


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return 0
    return int(value if sys.platform == "darwin" else value * 1024)


@dataclass
class Budget:
    per_pass: int = 32
    total: int = 128
    wall_s: float = 30.0
    started: float = field(default_factory=time.monotonic)
    started_rss: int = field(default_factory=_peak_rss_bytes)
    used_total: int = 0
    used_pass: int = 0
    exhausted: bool = False
    unevaluated: list[dict] = field(default_factory=list)

    def begin_pass(self) -> None:
        self.used_pass = 0

    def wall_exceeded(self) -> bool:
        return time.monotonic() - self.started > self.wall_s

    def rss_exceeded(self) -> bool:
        return _peak_rss_bytes() - self.started_rss > MAX_PEAK_BYTES

    def can_evaluate(self) -> bool:
        if self.used_total >= self.total or self.used_pass >= self.per_pass:
            self.exhausted = True
            return False
        if self.wall_exceeded() or self.rss_exceeded():
            self.exhausted = True
            return False
        return True

    def can_discover(self) -> bool:
        """Bound discovery even when no evaluation is charged for a candidate."""
        if self.wall_exceeded() or self.rss_exceeded():
            self.exhausted = True
            return False
        return True

    def charge(self) -> None:
        self.used_total += 1
        self.used_pass += 1

    def mark(self, candidate_id: str, pass_name: str, members, reason: str) -> None:
        self.unevaluated.append({"candidate_id": candidate_id, "pass": pass_name,
                                 "member_ops": list(members), "reason": reason})




@dataclass(frozen=True)
class PassReport:
    stage: str
    before_ms: float
    after_ms: float
    before_hbm: float
    after_hbm: float
    before_launches: int
    after_launches: int
    candidate_ids: tuple[str, ...]
    applied: bool
    rejected: tuple[str, ...]
    per_pass_budget: int = 32
    per_pass_used: int = 0

    @property
    def delta_ms(self) -> float:
        return self.before_ms - self.after_ms

    @property
    def delta_hbm(self) -> float:
        return self.before_hbm - self.after_hbm

    @property
    def delta_launches(self) -> int:
        return self.before_launches - self.after_launches

    def record(self) -> dict:
        return {"stage": self.stage, "before_ms": self.before_ms,
                "after_ms": self.after_ms, "delta_ms": self.delta_ms,
                "delta_hbm": self.delta_hbm, "delta_launches": self.delta_launches,
                "candidate_ids": list(self.candidate_ids), "applied": self.applied,
                "rejected": list(self.rejected),
                "per_pass_budget": self.per_pass_budget,
                "per_pass_used": self.per_pass_used}


@dataclass(frozen=True)
class Rejection:
    candidate_id: str
    pass_name: str
    member_ops: tuple[str, ...]
    reasons: tuple[str, ...]
    detail: str

    def record(self) -> dict:
        return {"candidate_id": self.candidate_id, "pass": self.pass_name,
                "member_ops": list(self.member_ops), "reasons": list(self.reasons),
                "detail": self.detail}


@dataclass
class OptimizationReport:
    graph_rows: int
    graph_operators: int
    hardware: HardwareProfile
    baseline: EngineEvaluation
    final: EngineEvaluation
    baseline_plan: tuple[PlanNode, ...]
    final_plan: tuple[PlanNode, ...]
    passes: tuple[PassReport, ...]
    rejections: tuple[Rejection, ...]
    applied: tuple[dict, ...]
    sol: dict
    capture: dict
    evaluations: int
    budget: int
    budget_exhausted: bool
    unevaluated: tuple[dict, ...] = ()
    per_pass_budget: int = 32
    provenance: str = "typed_projection_engine"
    canonical_plan: CanonicalPlan | None = None
    dag_edges: tuple[dict, ...] = ()

    @property
    def waterfall(self) -> tuple[dict, ...]:
        rows = [{"id": "baseline_unoptimized", "kind": "baseline",
                 "duration_ms": self.baseline.duration_ms,
                 "incremental_saved_ms": None, "basis": "engine_selected_plan",
                 "delta_hbm": 0.0, "delta_launches": 0}]
        for row in self.passes:
            rows.append({"id": row.stage, "kind": "accepted_pass" if row.applied else "pass",
                         "duration_ms": row.after_ms,
                         "incremental_saved_ms": row.delta_ms,
                         "basis": "engine_selected_plan",
                         "delta_hbm": row.delta_hbm,
                         "delta_launches": row.delta_launches})
        rows.append({"id": "lower_bound_sol", "kind": "lower_bound",
                     "duration_ms": self.sol["duration_ms"],
                     "incremental_saved_ms": None, "basis": "necessary_lower_bound",
                     "delta_hbm": None, "delta_launches": None})
        return tuple(rows)

    def dag(self) -> dict:
        edges = list(self.dag_edges) if self.dag_edges else [
            dict(edge) for edge in effective_edges(self.final_plan)]
        nodes = []
        for node in self.final_plan:
            nodes.append({"id": node.node_id, "kind": node.kind, "stream": node.stream,
                          "members": list(node.args), "recipe": node.recipe_id,
                          "work": dict(node.work), "hbm_read": node.hbm_read,
                          "hbm_write": node.hbm_write, "retained": list(node.retained),
                          "priority": node.priority, "caps": dict(node.caps),
                          "phase": node.phase,
                          "numerical_policy": node.numerical_policy})
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> dict:
        return {"status": "completed" if self.final.status == "completed" else "partial",
                "provenance": self.provenance,
                "rows": self.graph_rows, "operators": self.graph_operators,
                "baseline_ms": self.baseline.duration_ms,
                "final_ms": self.final.duration_ms,
                "baseline_hbm": self.baseline.hbm_bytes,
                "final_hbm": self.final.hbm_bytes,
                "baseline_launches": self.baseline.launches,
                "final_launches": self.final.launches,
                "passes": [p.record() for p in self.passes],
                "waterfall": list(self.waterfall), "sol": self.sol,
                "capture": self.capture, "rejections": [r.record() for r in self.rejections],
                "applied": list(self.applied), "dag": self.dag(),
                "engine_evaluations": self.evaluations,
                "evaluation_budget": self.budget, "budget_exhausted": self.budget_exhausted,
                "unevaluated": list(self.unevaluated),
                "per_pass_budget": self.per_pass_budget}




def _canonical_nodes(plan: CanonicalPlan) -> tuple[PlanNode, ...]:
    graph = plan.graph
    nodes = []
    for op in plan.ops:
        node = graph.ops[op.op_id]
        metadata = dict(op.metadata)
        family = op.family
        kind = ("prefetch" if family == "prefetch" else
                "release" if family == "release" else
                "event" if family == "event" else "kernel")
        priority = int(metadata.get("prefetch_priority", 0) or 0)
        caps = dict(metadata.get("prefetch_caps", {}) or {})
        prefetch_alloc = None
        free_alloc = None
        if kind == "prefetch":
            units = tuple(range(int(metadata.get("prefetch_units", 1) or 1)))
            size = int(metadata.get("prefetch_bytes", int(op.hbm_bytes)))
            prefetch_alloc = (f"{op.op_id}:reserve", "l1",
                              int(math.ceil(size / max(1, len(units)))), units)
        if kind == "release":
            units = tuple(range(int(metadata.get("prefetch_release_units", 1) or 1)))
            size = int(metadata.get("prefetch_release_bytes", 0))
            free_alloc = (f"{metadata.get('prefetch_release_for', op.op_id)}:reserve",
                          "l1", int(math.ceil(size / max(1, len(units)))), units)
        nodes.append(PlanNode(
            node_id=op.op_id,
            work=tuple(sorted((role, float(amount))
                              for role, amount in node.work)),
            deps=tuple(op.deps), stream=op.stream or "main", kind=kind,
            args=(op.op_id,), recipe_id=op.recipe, phase=op.phase,
            hbm_read=float(op.hbm_bytes), hbm_write=float(op.mte3_bytes),
            collective=node.collective, state_effect=node.state_effect,
            priority=priority, caps=tuple(sorted(caps.items())),
            prefetch_for=str(metadata.get("prefetch_target", "")),
            prefetch_alloc=prefetch_alloc, free_alloc=free_alloc,
            pointer_stable=bool(metadata.get("pointer_stable", False))
            and not bool(metadata.get("unstable_pointer")
                         or metadata.get("pointer_unstable")),
            capture_dynamic=bool(any(metadata.get(key) for key in (
                "dynamic_shape", "dynamic_dims", "shape_is_dynamic",
                "graph_dynamic_shape", "dynamic_shape_dims", "shape_dynamic",
                "symbolic_shape", "dynamic_shape_symbols"))),
            capture_ineligible=bool(metadata.get("capture_ineligible")),
            numerical_policy=node.numerical_policy))
    return tuple(nodes)


def _canonical_eval(result: dict, nodes: tuple[PlanNode, ...]) -> EngineEvaluation:
    timeline = timeline_by_op(result)
    completed = {op_id: float(row["complete_ms"]) for op_id, row in timeline.items()}
    started = {op_id: float(row["ready_ms"]) for op_id, row in timeline.items()}
    totals: dict[str, float] = {}
    for node in nodes:
        for role, amount in node.work:
            totals[role] = totals.get(role, 0.0) + float(amount)
    work = result.get("execution", {}).get("work", {})
    return EngineEvaluation(
        "completed", float(result["duration_ms"]), len(result.get("timeline", ())),
        float(work.get("hbm_bytes", totals.get("hbm", 0.0))),
        float(work.get("flops", totals.get("cube", 0.0))),
        float(work.get("vector_ops", totals.get("vector", 0.0))),
        tuple(sorted(started.items())), tuple(sorted(completed.items())), "",
        (), None, tuple(sorted((k, float(v)) for k, v in totals.items())))


def _canonical_dag(plan: CanonicalPlan, result: dict | None = None) -> tuple[dict, ...]:
    """Effective DAG in the exact execution order the Engine evaluated.

    Derives same-stream and group/event order directly from the executed
    canonical timeline (`result` or `plan.evaluate()`), so both the lazy FIFO
    path (`aa -> cc -> bb`) and the detailed kernel-group submission path
    (`aa -> bb -> cc`) export the exact edges observed by ``Execution``.
    """
    from mindie_sim.prediction.engine_plan import _stable_topological
    nodes = _canonical_nodes(plan)
    by_id = {node.node_id: node for node in nodes}
    actual = result if result is not None else plan.evaluate()
    timeline = [row for row in actual.get("timeline", ())
                if row.get("replica", 0) == 0 and row.get("op_id") in by_id]
    seen_ids = {row["op_id"] for row in timeline}
    fallback = [n for n in _stable_topological(nodes, plan.external_inputs)
                if n.node_id not in seen_ids]
    order = [by_id[row["op_id"]] for row in timeline] + fallback
    row_by_id = {row["op_id"]: row for row in timeline}
    metadata = {op.op_id: dict(op.metadata) for op in plan.ops}
    stages = {op.op_id: int(op.stage) for op in plan.ops}
    edges = []
    tails: dict[tuple[int, int, str], str] = {}
    for node in order:
        row = row_by_id.get(node.node_id, {})
        actual = set()
        for raw_pred in row.get("predecessors", ()):
            actual.add(raw_pred.split(":", 1)[1]
                       if raw_pred.startswith("r0:") else raw_pred)
        for dep in node.deps:
            if row and dep in by_id and dep not in actual:
                # A declared label is not an executed dependency.  A trial
                # prefetch that has not been joined must not export a join.
                continue
            kind = "prefetch_join" if dep.startswith("prefetch_W_") else "data"
            edges.append({"from": dep, "to": node.node_id, "kind": kind})
        for raw_pred in row.get("predecessors", ()):
            pred = raw_pred.split(":", 1)[1] if raw_pred.startswith("r0:") else raw_pred
            if pred in by_id and pred not in node.deps:
                pred_node = by_id[pred]
                if (pred_node.stream or "main") != (node.stream or "main"):
                    edges.append({"from": pred, "to": node.node_id, "kind": "event_cross_stream"})
                elif not (pred_node.kind == "kernel" and node.kind == "kernel"):
                    edges.append({"from": pred, "to": node.node_id, "kind": "group_order"})
        if node.kind == "kernel":
            # Execution's physical stream identity is (replica, stage, stream);
            # two pipeline stages that both say "main" are independent streams.
            row = row_by_id.get(node.node_id, {})
            stream = (int(row.get("replica", 0)),
                      int(row.get("stage", stages.get(node.node_id, 0))),
                      node.stream or "main")
            if stream in tails:
                edges.append({"from": tails[stream], "to": node.node_id,
                              "kind": "stream_order"})
            tails[stream] = node.node_id
        if node.kind == "release":
            for dep in node.deps:
                edges.append({"from": dep, "to": node.node_id,
                              "kind": "reservation_release"})
            prefetch_id = metadata.get(node.node_id, {}).get(
                "prefetch_release_for")
            if prefetch_id is None:
                prefetch_id = next((dep for dep in node.deps
                                    if dep.startswith("prefetch_W_")), None)
            if prefetch_id is not None and prefetch_id in node.deps:
                edges.append({"from": prefetch_id, "to": node.node_id,
                              "kind": "pool_release"})
    unique, seen = [], set()
    for edge in edges:
        key = (edge["from"], edge["to"], edge["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return tuple(unique)
