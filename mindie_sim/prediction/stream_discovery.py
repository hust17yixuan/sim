"""Budget-bounded typed graph indexing and stream-region discovery.

[Role] Build one forward index over a typed optimizer graph and discover legal
  multi-stream regions over a ``PlanNode`` DAG using iterative bitsets.
[Invariants] One name-to-index map resolves dependencies in constant time.
  Logical inner-work counts and every 64-tick wall/RSS poll are preserved.
  Packed single edges and deferred contiguous ancestry avoid growing chain
  bitsets; only joins consume ancestry, and only forks consume descendants.
  Fixed-work slices stop before polling boundaries; redundant sorted/topological
  passes and unused descendant work retain their original tick charges.
  Join lineage isolation uses linear prefix/suffix ORs. Regions already found
  survive exhaustion, and growth is capped before construction.
[Timeouts & Locks] Caller wall/RSS/inner-work budgets only; no I/O or locks.
[Side-Effect Policy] Pure functions over immutable PlanNode/TypedGraph records.
"""
from __future__ import annotations

from array import array

from mindie_sim.prediction.fusion_graph import TypedGraph
from mindie_sim.prediction.plan_model import PlanNode

from .optimizer_report import Budget

_TICK_MASK = 63
_MAX_REGION_MEMBERS = 8


class _BudgetStop(Exception):
    """Internal unwind used only when a discovery budget expires."""


class _Ticker:
    """One inner-work counter shared by every loop of a discovery call."""
    __slots__ = ("budget", "ticks")

    def __init__(self, budget: Budget | None):
        self.budget = budget
        self.ticks = 0

    def tick(self) -> None:
        """Count one inner-work unit; poll wall/RSS every 64 ticks."""
        self.ticks += 1
        if self.budget is not None and not (self.ticks & _TICK_MASK):
            if not self.budget.can_discover():
                raise _BudgetStop

    def advance(self, count: int) -> None:
        """Charge elided work, retaining every intervening polling boundary."""
        stop = self.ticks + count
        boundary = self.ticks + (_TICK_MASK - (self.ticks & _TICK_MASK))
        while boundary < stop:
            self.ticks = boundary
            self.tick()
            boundary = self.ticks + _TICK_MASK
        self.ticks = stop

    def blocks(self, count: int):
        """Fixed-work slices; poll before the first item at a tick boundary."""
        start = 0
        while start < count:
            self.tick()
            stop = min(count, start + 64 - (self.ticks & _TICK_MASK))
            self.ticks += stop - start - 1
            yield start, stop
            start = stop


def index_graph(graph: TypedGraph, budget: Budget | None = None
                ) -> tuple[dict[str, tuple[str, ...]] | None,
                           dict[str, list[str]] | None]:
    """One budget-bounded forward index in ``TypedGraph.successors`` order.

    Returns ``(successors, consumers)``, or ``(None, None)`` on expiry.
    """
    ticker = _Ticker(budget)
    consumers: dict[str, list[str]] = {}
    try:
        for op in graph.ops.values():
            ticker.tick()
            for value_id in op.inputs:
                ticker.tick()
                consumers.setdefault(value_id, []).append(op.op_id)
        dependency: dict[str, list[str]] = {}
        for op_id in graph.ops:
            ticker.tick()
            dependency[op_id] = []
        for other in graph.ops.values():
            ticker.tick()
            for dep in other.deps:
                ticker.tick()
                if dep in dependency and dep != other.op_id:
                    dependency[dep].append(other.op_id)
        successors: dict[str, tuple[str, ...]] = {}
        for op_id, op in graph.ops.items():
            ticker.tick()
            found: list[str] = []
            seen: set[str] = set()
            for value_id in op.outputs:
                ticker.tick()
                for consumer in consumers.get(value_id, ()):
                    ticker.tick()
                    if consumer != op_id and consumer not in seen:
                        seen.add(consumer)
                        found.append(consumer)
            for successor in dependency.get(op_id, ()):
                ticker.tick()
                if successor not in seen:
                    seen.add(successor)
                    found.append(successor)
            successors[op_id] = tuple(found)
        return successors, consumers
    except _BudgetStop:
        return None, None


def adjacency(graph: TypedGraph, budget: Budget | None = None
              ) -> dict[str, tuple[str, ...]]:
    """Pre-indexed forward neighbours, or ``{}`` on budget exhaustion."""
    successors, _consumers = index_graph(graph, budget)
    return successors or {}


def _names(plan: tuple[PlanNode, ...], regions: list[tuple[int, ...]],
           ticker: _Ticker | None = None) -> tuple[tuple[str, ...], ...]:
    """Convert index regions to plan-order names, preserving uniqueness.

    Member conversion ticks the shared counter; if a wall/RSS budget expires
    exactly while naming the regions already found, the converted prefix is
    returned instead of discarding discoveries.
    """
    converted: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for region in regions:
        row: list[str] = []
        for index in region:
            if ticker is not None:
                try:
                    ticker.tick()
                except _BudgetStop:
                    if row:
                        name_row = tuple(row)
                        if name_row not in seen:
                            converted.append(name_row)
                    return tuple(converted)
            row.append(plan[index].node_id)
        name_row = tuple(row)
        if name_row not in seen:
            seen.add(name_row)
            converted.append(name_row)
    return tuple(converted)


def _materialize(mask: int, ticker: _Ticker) -> tuple[int, ...]:
    """Plan-order indices of a region bitmask, one tick per member."""
    found: list[int] = []
    while mask:
        ticker.tick()
        low = mask & -mask
        found.append(low.bit_length() - 1)
        mask ^= low
    found.sort()
    return tuple(found)


def stream_regions(plan: tuple[PlanNode, ...], max_streams: int,
                   budget: Budget | None = None) -> tuple[tuple[str, ...], ...]:
    """Bounded stream-region discovery with iterative bitset ancestry.

    Reachability is one topological bitset pass.  Join lineage isolation uses
    prefix/suffix ORs in ``O(D)``, so ``D`` incoming deps never cost ``D^2``.
    A fork candidate is the branch's descendant set minus the rejoin node, so a
    wide star never materializes a full-plan region per branch.  Exhaustion
    returns the regions found so far.
    """
    count = len(plan)
    if max_streams < 2 or not count:
        return ()
    ticker = _Ticker(budget)
    regions: list[tuple[int, ...]] = []
    try:
        name_to_idx: dict[str, int] = {}
        for start, stop in ticker.blocks(count):
            for position in range(start, stop):
                name_to_idx.setdefault(plan[position].node_id, position)
        # Packed indices contain no per-node object references; rows are sparse.
        single_child = array('i', (-1,)) * count
        forks: dict[int, list[int]] = {}
        ticker.advance(count)
        single_parent = array('i', (-1,)) * count
        multi_deps: dict[int, tuple[int, ...]] = {}
        need_anc = bytearray(count)
        forward = True
        has_fork = has_join = False
        edge_count = 0
        ticks = ticker.ticks
        name_get = name_to_idx.get
        for position, node in enumerate(plan):
            ticks += 1
            if not (ticks & _TICK_MASK):
                ticker.ticks = ticks - 1
                ticker.tick()
            dependencies = node.deps
            if len(dependencies) <= 1:
                if not dependencies:
                    continue
                ticks += 1
                if not (ticks & _TICK_MASK):
                    ticker.ticks = ticks - 1
                    ticker.tick()
                parent = name_get(dependencies[0])
                if parent is None or parent == position:
                    continue
                single_parent[position] = parent
                child = single_child[parent]
                if child == -1:
                    single_child[parent] = position
                elif child >= 0:
                    forks[parent] = [child, position]
                    single_child[parent] = -2
                    need_anc[parent] = 1
                    has_fork = True
                else:
                    forks[parent].append(position)
                edge_count += 1
                forward = forward and parent < position
                continue
            found: list[int] = []
            seen: set[int] = set()
            for dep in dependencies:
                ticks += 1
                if not (ticks & _TICK_MASK):
                    ticker.ticks = ticks - 1
                    ticker.tick()
                parent = name_get(dep)
                if parent is None or parent == position or parent in seen:
                    continue
                seen.add(parent)
                found.append(parent)
                need_anc[parent] = 1
                child = single_child[parent]
                if child == -1:
                    single_child[parent] = position
                elif child >= 0:
                    forks[parent] = [child, position]
                    single_child[parent] = -2
                    has_fork = True
                else:
                    forks[parent].append(position)
                forward = forward and parent < position
            edge_count += len(found)
            multi_deps[position] = tuple(found)
            has_join = has_join or bool(found)
        ticker.ticks = ticks
        # Children were appended in plan order; sorting each row is redundant.
        ticker.advance(count)
        if forward:
            # Plan order already is topological. Charge the original ready scan,
            # Kahn node visits and edge visits, including every budget poll.
            ticker.advance(count)
            ticker.advance(count + edge_count)
            order = range(count)
        else:
            indegree = array('I', (0,)) * count
            ready: list[int] = []
            for index in range(count):
                ticker.tick()
                indegree[index] = (1 if single_parent[index] >= 0
                                   else len(multi_deps.get(index, ())))
                if not indegree[index]:
                    ready.append(index)
            order = []
            head = 0
            while head < len(ready):
                ticker.tick()
                node = ready[head]
                head += 1
                order.append(node)
                child = single_child[node]
                if child == -1:
                    continue
                for child in (forks[node] if child == -2 else (child,)):
                    ticker.tick()
                    indegree[child] -= 1
                    if not indegree[child]:
                        ready.append(child)
            if len(order) != count:
                return ()
        for _start, _stop in ticker.blocks(count):
            pass  # Preserve the former bit-table work and physical tick calls.
        if has_join:
            # A deferred node represents ancestors[lo] plus [lo, node).
            # Materialized endpoints reset lo to themselves. Missing bases
            # are roots with no ancestors; joins and index jumps flush runs.
            ancestors: dict[int, int] = {}
            run_lo = array('i', (0,)) * count
        ticks = ticker.ticks
        for node in order:
            ticks += 1
            if not (ticks & _TICK_MASK):
                ticker.ticks = ticks - 1
                ticker.tick()
            parent = single_parent[node]
            if parent >= 0:
                ticks += 1
                if not (ticks & _TICK_MASK):
                    ticker.ticks = ticks - 1
                    ticker.tick()
                if has_join:
                    lo = run_lo[parent]
                    if node == parent + 1 and not need_anc[node]:
                        run_lo[node] = lo
                    else:
                        # The deferred run is contiguous on this dependency
                        # path, even when Kahn visits other paths between it.
                        ancestors[node] = (ancestors.get(lo, 0)
                                           | (((1 << (parent - lo + 1)) - 1) << lo))
                        run_lo[node] = node
            elif has_join:
                run_lo[node] = node
                parents = multi_deps.get(node, ())
                if not parents:
                    continue
                ticker.ticks = ticks
                mask = 0
                for parent in parents:
                    ticker.tick()
                    mask |= ancestors.get(parent, 0) | (1 << parent)
                ticks = ticker.ticks
                ancestors[node] = mask
        ticker.ticks = ticks
        if has_fork:
            descendants = [0] * count
            for node in reversed(order):
                ticker.tick()
                mask = 0
                child = single_child[node]
                if child == -1:
                    continue
                for child in (forks[node] if child == -2 else (child,)):
                    ticker.tick()
                    mask |= (1 << child) | descendants[child]
                descendants[node] = mask
        else:
            # Descendant masks are consumed only by fork candidates.
            ticker.advance(count + edge_count)
        cap = (max(budget.total * 2, budget.per_pass * 2, 64)
               if budget is not None else None)

        def at_cap() -> bool:
            if cap is None or len(regions) < cap:
                return False
            if budget is not None:
                budget.exhausted = True
            return True

        def legal(region: tuple[int, ...]) -> bool:
            if not region or len(region) > _MAX_REGION_MEMBERS:
                return False
            for index in region:
                ticker.tick()
                node = plan[index]
                if node.kind != "kernel" or node.stream != "main" \
                        or node.collective or node.capture_boundary \
                        or node.state_effect:
                    return False
            return True

        ticks = ticker.ticks
        for join_idx, join in enumerate(plan):
            ticks += 1
            if not (ticks & _TICK_MASK):
                ticker.ticks = ticks - 1
                ticker.tick()
            if len(join.deps) < 2:
                continue
            incoming = multi_deps.get(join_idx, ())
            if not incoming:
                continue
            ticker.ticks = ticks
            lineages: list[int] = []
            for parent in incoming:
                ticker.tick()
                lineages.append(ancestors.get(parent, 0) | (1 << parent))
            width = len(lineages)
            prefix = [0] * (width + 1)
            suffix = [0] * (width + 1)
            for position in range(width):
                ticker.tick()
                prefix[position + 1] = prefix[position] | lineages[position]
            for position in range(width - 1, -1, -1):
                ticker.tick()
                suffix[position] = suffix[position + 1] | lineages[position]
            for position in range(width):
                ticker.tick()
                exclusive = lineages[position] & ~(prefix[position]
                                                   | suffix[position + 1])
                if exclusive.bit_count() > _MAX_REGION_MEMBERS:
                    continue
                region = _materialize(exclusive, ticker)
                if legal(region):
                    regions.append(region)
                    if at_cap():
                        return _names(plan, regions, ticker)
            ticks = ticker.ticks
        ticker.ticks = ticks
        if not has_fork:
            ticker.advance(count)
            return _names(plan, regions, ticker)
        for fork_idx in range(count):
            ticker.tick()
            branches = forks.get(fork_idx)
            if not branches:
                continue
            for branch in branches:
                ticker.tick()
                descendants_mask = descendants[branch]
                if not descendants_mask:
                    continue
                if descendants_mask.bit_count() > _MAX_REGION_MEMBERS + 1:
                    continue
                pending = descendants_mask
                while pending:
                    ticker.tick()
                    low = pending & -pending
                    join_idx = low.bit_length() - 1
                    pending ^= low
                    region_mask = descendants_mask & ~(1 << join_idx)
                    region = _materialize(region_mask, ticker)
                    if legal(region):
                        regions.append(region)
                        if at_cap():
                            return _names(plan, regions, ticker)
    except _BudgetStop:
        return _names(plan, regions, ticker)
    return _names(plan, regions, ticker)
