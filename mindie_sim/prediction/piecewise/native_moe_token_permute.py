"""R64 outgoing MoeTokenPermute leaf plus the shared MoE route/layout contract.

[Role] Expand the source-selected prefill outbound
  ``torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens)`` call
  from its real R64 caller, ``TokenDispatcherWithAll2AllV._dispatch_preprocess``,
  into the actual BF16 dense-row / INT32 top-k input and expanded-row output
  descriptors for the evaluated token geometry, and expose one typed common
  program fragment for that call.  The same module owns the small route/layout
  contract consumed by the sibling inbound ``MoeInitRoutingV3`` and histogram
  leaves: declared token->expert identities, integer per-expert counts,
  stable expert-ascending assignment order and exact offsets.
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the caller blob that is invariant across every source
  commit whose ``device_op.py`` is the R64 ``2a251259...`` blob.  M and the
  expanded row count ``M*k`` live only in the occurrence; ``k``, H, dtype,
  API, kernel name and output contract stay in the binding.  A route is an
  occurrence input: counts/ids are never synthesized from a balanced
  assumption unless the caller explicitly selects the balanced performance
  control.  No tensor is allocated, no pointer or symbol is fabricated, no
  device launch is claimed and no measurement is used to price or select work.
  The outbound TokenPermute is not ``npu_moe_init_routing_v2``; the latter is
  a different API and is deliberately not admitted here.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal geometry; unknown source/ABI stays absent or unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from mindie_sim.prediction.program import KernelProgram, scope_allocations
from mindie_sim.schema.common import ModelingError

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch_npu.npu_moe_token_permute"
ACLNN_API = "aclnnMoeTokenPermute"
KERNEL_NAME = "MoeInitRoutingV2"
PROFILER_NAME = "aclnnMoeTokenPermute_MoeInitRoutingV2_MoeInit"
SOURCE_LINES = (619, 623)
NUM_OUT_TOKENS_LINE = 647
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "sha256": "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5",
    "symbol": "TokenDispatcherWithAll2AllV._dispatch_preprocess",
    "revision": REVISION,
    "state": "task_artifact",
}
ROLE = "outbound_local_before_dispatch"
SOURCE_SELECTED = "prefill_local_outbound"
_DYNAMIC_CACHE: dict[tuple, dict] = {}

# ---------------------------------------------------------------------------
# Shared MoE route/layout and typed-work contract.
#
# These records are occurrence facts, not part of the native call identity.
# A declared token->expert identity list gives the exact stable assignment
# order; integer counts give the exact per-expert row blocks and offsets but
# not the identity order inside a block.  Missing route facts stay explicit;
# a balanced split exists only when the caller names the performance control.
# ---------------------------------------------------------------------------

ROUTE_SCHEMA = "mindie-sim/moe-route-layout/1"
ROUTE_BASIS_IDS = "declared_token_expert_ids"
ROUTE_BASIS_COUNTS = "declared_expert_token_counts"
ROUTE_BASIS_GEOMETRY = "geometry_routing_counts"
ROUTE_BASIS_BALANCED = "forced_balanced_performance_control"
ROUTE_BASIS_UNRESOLVED = "unresolved"
WORK_UNITS = {
    "cube": "flops", "vector": "ops", "hbm": "bytes", "mte1": "bytes",
    "mte2": "bytes", "mte3": "bytes", "fixpipe": "bytes", "link": "bytes",
}


@dataclass(frozen=True)
class MoeRoute:
    """One declared route view with forward order and its ABI inverse.

    ``forward[j]`` is the original assignment identity of the j-th sorted
    assignment; ``inverse[forward[j]] == j``.  The forward order drives
    expert blocks and data movement; the inverse is the ABI output used by
    unpermute/reorder consumers.  Token/slot identities are derived from the
    complete assignment identity, not from a token-only shortcut.
    """

    basis: str
    scope: str
    counts: tuple[int, ...] | None = None
    offsets: tuple[int, ...] | None = None
    forward: tuple[int, ...] | None = None
    inverse: tuple[int, ...] | None = None
    assignment_experts: tuple[int, ...] | None = None
    assignment_tokens: tuple[int, ...] | None = None
    assignment_slots: tuple[int, ...] | None = None
    identity_space: str = "not_declared"
    rows: int | None = None
    expert_range: int | None = None
    counts_complete: bool = False
    performance_control: bool = False
    token_identities: str = "not_declared"
    within_expert_order: str = "unresolved_no_token_identities"
    source: str = ""
    unresolved: tuple[str, ...] = ()

    @property
    def experts(self) -> int | None:
        """The complete expert/range width, not the observed distinct set."""
        return self.expert_range

    @property
    def active_experts(self) -> int | None:
        if self.counts is not None and self.counts_complete:
            return sum(1 for value in self.counts if value > 0)
        if self.assignment_experts:
            return len(set(self.assignment_experts))
        return None

    @property
    def observed_counts(self) -> tuple[tuple[int, int], ...]:
        """Contiguous observed expert runs; a lower bound, never a full range."""
        if not self.assignment_experts:
            return ()
        runs: list[tuple[int, int]] = []
        expert = self.assignment_experts[0]
        count = 1
        for value in self.assignment_experts[1:]:
            if value == expert:
                count += 1
            else:
                runs.append((expert, count))
                expert, count = value, 1
        runs.append((expert, count))
        return tuple(runs)

    @property
    def layout_digest(self) -> str:
        payload = json.dumps(
            {"counts": self.counts, "counts_complete": self.counts_complete,
             "expert_range": self.expert_range, "offsets": self.offsets,
             "forward": self.forward, "inverse": self.inverse,
             "assignment_experts": self.assignment_experts,
             "assignment_tokens": self.assignment_tokens,
             "assignment_slots": self.assignment_slots},
            sort_keys=True, separators=(",", ":"), default=str)
        return sha256(payload.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_SCHEMA,
            "basis": self.basis,
            "scope": self.scope,
            "rows": self.rows,
            "experts": self.expert_range,
            "expert_range": self.expert_range,
            "counts_complete": self.counts_complete,
            "observed_counts": [list(item) for item in self.observed_counts],
            "active_experts": self.active_experts,
            "performance_control": self.performance_control,
            "token_identities": self.token_identities,
            "within_expert_order": self.within_expert_order,
            "source": self.source,
            "unresolved": list(self.unresolved),
            "counts": list(self.counts) if self.counts is not None else None,
            "expert_offsets": (list(self.offsets)
                               if self.offsets is not None else None),
            "forward_assignment_order": (list(self.forward)
                                         if self.forward is not None else None),
            "inverse_assignment_order": (list(self.inverse)
                                         if self.inverse is not None else None),
            "assignment_experts": (list(self.assignment_experts)
                                   if self.assignment_experts is not None else None),
            "assignment_tokens": (list(self.assignment_tokens)
                                  if self.assignment_tokens is not None else None),
            "assignment_slots": (list(self.assignment_slots)
                                 if self.assignment_slots is not None else None),
            "assignment_identity": self.identity_space,
            "layout_digest": self.layout_digest,
        }

def route_ints(value: Any, name: str, *, length: int | None = None,
               maximum: int | None = None) -> tuple[int, ...] | None:
    """Validate one non-negative integer route field without coercion."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ModelingError(f"moe_route_{name}_requires_sequence")
    result: list[int] = []
    for item in value:
        if type(item) is not int or item < 0 or (
                maximum is not None and item > maximum):
            raise ModelingError(f"moe_route_{name}_invalid:{item!r}")
        result.append(item)
    if length is not None and len(result) != int(length):
        raise ModelingError(
            f"moe_route_{name}_length_mismatch:{len(result)}!={int(length)}")
    return tuple(result)


def route_control(value: Any) -> str | None:
    """Only the explicit balanced performance control is admitted."""
    text = str(value or "").strip().lower()
    if text in {"", "none"}:
        return None
    if text == "balanced":
        return "balanced"
    raise ModelingError(f"moe_route_control_unsupported:{value}")


def route_inputs(compiled: Any, attrs: Any) -> dict[str, Any]:
    """Read declared route inputs once: typed attrs first, then the source theta."""
    theta = getattr(getattr(compiled, "operator", None), "theta", None)
    result: dict[str, Any] = {}
    for name in ("router_topk_ids", "router_expert_token_counts",
                 "router_local_expert_ids", "router_expert_placement",
                 "router_route_control"):
        value = dict(attrs or {}).get(name)
        if value is None and theta is not None:
            value = theta.find(name, None)
        result[name] = value
    return result


def counts_from_ids(ids: Any, experts: int, *, name: str = "ids") -> tuple[int, ...]:
    values = route_ints(ids, name, maximum=int(experts) - 1)
    counts = [0] * int(experts)
    for expert in values or ():
        counts[expert] += 1
    return tuple(counts)


def offsets_from_counts(counts: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for value in counts:
        offsets.append(offsets[-1] + int(value))
    return tuple(offsets)


def _inverse_order(forward: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(forward)
    for position, original in enumerate(forward):
        inverse[original] = position
    return tuple(inverse)


def layout_from_ids(ids: Any, experts: int) -> tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...],
        tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Exact grouped layout for one local-expert id per original row.

    Original assignment identity is the row index.  ``forward`` is the sorted
    order; ``inverse`` is the ABI ``expanded_row_idx``/GATHER output.
    """
    values = route_ints(ids, "local_expert_ids", maximum=int(experts) - 1)
    assert values is not None
    forward = tuple(sorted(range(len(values)),
                           key=lambda index: (values[index], index)))
    inverse = _inverse_order(forward)
    assignment_experts = tuple(values[index] for index in forward)
    assignment_tokens = forward
    assignment_slots = (0,) * len(forward)
    counts = counts_from_ids(values, experts, name="local_expert_ids")
    return (counts, offsets_from_counts(counts), forward, inverse,
            assignment_experts, assignment_tokens, assignment_slots)


def layout_from_topk(topk_ids: Any, tokens: int, top_k: int,
                     experts: int) -> tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...],
        tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Exact grouped layout over the complete ``token*top_k+slot`` identity."""
    if isinstance(topk_ids, (str, bytes)) or not isinstance(
            topk_ids, (list, tuple)):
        raise ModelingError("moe_route_topk_ids_requires_sequence")
    if len(topk_ids) != int(tokens):
        raise ModelingError(
            "moe_route_topk_ids_token_count_mismatch:"
            f"{len(topk_ids)}!={int(tokens)}")
    rows: list[tuple[int, int]] = []
    for token, row in enumerate(topk_ids):
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
            raise ModelingError(
                f"moe_route_topk_ids_row_requires_sequence:{token}")
        if len(row) != int(top_k):
            raise ModelingError(
                f"moe_route_topk_ids_top_k_mismatch:{token}:"
                f"{len(row)}!={int(top_k)}")
        for slot, expert in enumerate(row):
            if type(expert) is not int or not 0 <= expert < int(experts):
                raise ModelingError(
                    f"moe_route_topk_ids_expert_invalid:{token}:{expert!r}")
            rows.append((expert, token * int(top_k) + slot))
    rows.sort()
    forward = tuple(original for _expert, original in rows)
    inverse = _inverse_order(forward)
    assignment_experts = tuple(expert for expert, _original in rows)
    assignment_tokens = tuple(original // int(top_k) for original in forward)
    assignment_slots = tuple(original % int(top_k) for original in forward)
    counts = counts_from_ids(assignment_experts, experts,
                             name="topk_assignment_experts")
    return (counts, offsets_from_counts(counts), forward, inverse,
            assignment_experts, assignment_tokens, assignment_slots)



def balanced_counts(rows: int, experts: int) -> tuple[int, ...]:
    """The explicit performance control's deterministic integer split."""
    rows, experts = int(rows), int(experts)
    if rows < 0 or experts < 1:
        raise ModelingError("moe_route_balanced_geometry_invalid")
    base, extra = divmod(rows, experts)
    return tuple(base + (1 if index < extra else 0) for index in range(experts))


def _permutation(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if sorted(values) != list(range(len(values))):
        raise ModelingError(f"moe_route_{name}_not_a_permutation")
    return values


def make_route(*, basis: str, scope: str, counts: Any = None,
               rows: int | None = None, source: str = "",
               forward: Any = None, inverse: Any = None,
               assignment_experts: Any = None,
               assignment_tokens: Any = None, assignment_slots: Any = None,
               identity_space: str = "not_declared",
               expert_range: int | None = None,
               counts_complete: bool | None = None,
               performance_control: bool = False,
               unresolved: tuple[str, ...] = ()) -> MoeRoute:
    values = None if counts is None else route_ints(
        counts, "expert_token_counts")
    if values is not None and not values:
        raise ModelingError("moe_route_expert_token_counts_empty")
    if values is not None:
        total = sum(values)
        if total < 1:
            raise ModelingError("moe_route_expert_token_counts_zero_rows")
        if expert_range is None:
            expert_range = len(values)
        elif len(values) != int(expert_range):
            raise ModelingError(
                "moe_route_expert_token_counts_scope_mismatch:"
                f"{len(values)}!={int(expert_range)}")
        if counts_complete is None:
            counts_complete = True
        offsets = offsets_from_counts(values)
    else:
        if counts_complete is None:
            counts_complete = False
        offsets = None
        if forward is not None:
            total = len(tuple(forward))
        elif rows is not None:
            total = int(rows)
        else:
            raise ModelingError("moe_route_rows_unresolved")
    if rows is not None and int(rows) != total:
        raise ModelingError(f"moe_route_rows_mismatch:{total}!={int(rows)}")
    order = None
    reverse = None
    assignment = None
    tokens = None
    slots = None
    if forward is not None:
        order = _permutation(route_ints(forward, "forward_assignment_order",
                                       length=total) or (), "forward")
        reverse = _permutation(route_ints(inverse, "inverse_assignment_order",
                                          length=total) or (), "inverse")
        if any(reverse[original] != position
               for position, original in enumerate(order)):
            raise ModelingError("moe_route_forward_inverse_mismatch")
        assignment = route_ints(assignment_experts, "assignment_experts",
                                length=total,
                                maximum=(int(expert_range) - 1
                                         if expert_range is not None else None))
        if assignment is None:
            raise ModelingError("moe_route_assignment_experts_required_with_order")
        if any(left > right for left, right in zip(assignment, assignment[1:])):
            raise ModelingError("moe_route_assignment_order_not_expert_ascending")
        if values is not None:
            domain = int(expert_range if expert_range is not None else len(values))
            if counts_from_ids(assignment, domain,
                               name="topk_assignment_experts") != values:
                raise ModelingError("moe_route_assignment_disagrees_with_counts")
        tokens = route_ints(assignment_tokens, "assignment_tokens",
                            length=total)
        slots = route_ints(assignment_slots, "assignment_slots",
                           length=total)
    return MoeRoute(
        basis=basis, scope=scope, counts=values, offsets=offsets,
        forward=order, inverse=reverse, assignment_experts=assignment,
        assignment_tokens=tokens, assignment_slots=slots,
        identity_space=identity_space,
        rows=total, expert_range=expert_range,
        counts_complete=bool(counts_complete),
        performance_control=bool(performance_control),
        token_identities=("explicit_assignment"
                          if order is not None else "not_declared"),
        within_expert_order=("explicit_stable_expert_ascending"
                             if order is not None
                             else "unresolved_no_token_identities"),
        source=source, unresolved=tuple(dict.fromkeys(unresolved)))


def unresolved_route(*, scope: str, rows: int | None, experts: int | None,
                     source: str, reason: str,
                     performance_control: bool = False) -> MoeRoute:
    return MoeRoute(
        basis=ROUTE_BASIS_UNRESOLVED, scope=scope, rows=rows,
        expert_range=experts, performance_control=bool(performance_control),
        source=source, unresolved=tuple(dict.fromkeys((reason,))))


def normalize_route(
    *,
    declared: Any,
    geometry_counts: Any,
    scope: str,
    experts: int,
    rows: int,
    identity: str = "",
    tokens: int = 0,
    top_k: int = 0,
    geometry_source: str = "",
    extra_unresolved: tuple[str, ...] = (),
) -> MoeRoute:
    """Normalize one complete global route/contract fact set."""
    declared = dict(declared or {})
    control = route_control(declared.get("router_route_control"))
    extra = list(extra_unresolved)
    experts, rows = int(experts), int(rows)

    def valid_counts(raw: Any, name: str) -> tuple[int, ...] | None:
        values = route_ints(raw, name)
        if values is None:
            return None
        if len(values) != experts:
            raise ModelingError(
                f"moe_route_{name}_scope_mismatch:{len(values)}!={experts}")
        if sum(values) != rows:
            raise ModelingError(
                f"moe_route_{name}_rows_mismatch:{sum(values)}!={rows}")
        return values

    declared_counts = valid_counts(
        declared.get("router_expert_token_counts"), "expert_token_counts")
    forward = inverse = assignment = assignment_tokens = assignment_slots = None
    identity_counts = None
    identity_source = ""
    identity_unresolved: list[str] = []
    if identity == "topk_ids":
        raw = declared.get("router_topk_ids")
        if raw is not None:
            invalid_shape = (
                isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple))
                or len(raw) != int(tokens) or any(
                    isinstance(row, (str, bytes))
                    or not isinstance(row, (list, tuple))
                    or len(row) != int(top_k) for row in raw))
            if invalid_shape:
                identity_unresolved.append(
                    "router_topk_ids_do_not_match_evaluated_token_geometry")
            else:
                (identity_counts, _o, forward, inverse, assignment,
                 assignment_tokens, assignment_slots) = layout_from_topk(
                    raw, int(tokens), int(top_k), experts)
                identity_source = "router_topk_ids"
    elif identity:
        raise ModelingError(f"moe_route_identity_kind_unknown:{identity}")

    if (identity_counts is not None and declared_counts is not None
            and identity_counts != declared_counts):
        raise ModelingError("moe_route_ids_counts_conflict")
    counts = None
    basis = ROUTE_BASIS_UNRESOLVED
    source = ""
    if identity_counts is not None:
        counts = identity_counts
        basis = ROUTE_BASIS_IDS
        source = identity_source
    elif declared_counts is not None:
        counts = declared_counts
        basis = ROUTE_BASIS_COUNTS
        source = "router_expert_token_counts"

    geometry_values = None
    if geometry_counts is not None:
        geometry_values = route_ints(geometry_counts, "geometry_routing_counts")
        if geometry_values is not None and (
                len(geometry_values) != experts
                or sum(geometry_values) != rows):
            extra.append("geometry_routing_counts_scope_mismatch")
            geometry_values = None
    if geometry_values is not None:
        if counts is not None:
            if geometry_values != counts:
                extra.append("geometry_route_disagrees_with_selected_route")
        else:
            counts = geometry_values
            basis = ROUTE_BASIS_GEOMETRY
            source = geometry_source or "geometry_routing_counts"

    if control == "balanced":
        if counts is not None:
            raise ModelingError(
                "moe_route_actual_route_and_balanced_control_conflict")
        counts = balanced_counts(rows, experts)
        basis = ROUTE_BASIS_BALANCED
        source = "router_route_control=balanced"
        return make_route(
            basis=basis, scope=scope, counts=counts, rows=rows,
            source=source, expert_range=experts, counts_complete=True,
            performance_control=True,
            unresolved=tuple(extra + identity_unresolved))

    if counts is None:
        return MoeRoute(
            basis=ROUTE_BASIS_UNRESOLVED, scope=scope, rows=rows,
            expert_range=experts, source="no_route_facts",
            unresolved=tuple(dict.fromkeys(
                extra + identity_unresolved
                + ["expert_token_counts_unresolved"])))
    return make_route(
        basis=basis, scope=scope, counts=counts, rows=rows, source=source,
        forward=forward, inverse=inverse, assignment_experts=assignment,
        assignment_tokens=assignment_tokens, assignment_slots=assignment_slots,
        identity_space=("flattened_token_slot" if forward is not None
                        else "not_declared"),
        expert_range=experts, counts_complete=True,
        unresolved=tuple(extra + identity_unresolved))


def normalize_local_route(
    *,
    declared: Any,
    geometry_counts: Any,
    global_experts: int,
    local_range: int | None,
    rows: int | None,
    geometry_source: str = "",
    extra_unresolved: tuple[str, ...] = (),
) -> MoeRoute:
    """Normalize one inbound local route without inferring a range from ids.

    A complete local count vector and an explicit range both identify the
    domain.  Observed ids alone identify assignments, not the full local
    range, so a count output and its write work stay unresolved until a range
    is declared.  A local range larger than the global expert count is
    rejected.
    """
    declared = dict(declared or {})
    control = route_control(declared.get("router_route_control"))
    extra = list(extra_unresolved)
    global_experts = int(global_experts)
    if global_experts < 1:
        raise ModelingError("moe_route_global_experts_invalid")
    raw_ids = declared.get("router_local_expert_ids")
    raw_counts = declared.get("router_expert_token_counts")
    ids = route_ints(raw_ids, "local_expert_ids") if raw_ids is not None else None
    declared_counts = (route_ints(raw_counts, "expert_token_counts")
                       if raw_counts is not None else None)
    geometry_values = (route_ints(geometry_counts, "geometry_routing_counts")
                       if geometry_counts is not None else None)

    # Authoritative rows: explicit/received ledger, declared counts, then ids.
    row_values: list[int] = []
    if rows is not None:
        row_values.append(int(rows))
    if declared_counts is not None:
        if not declared_counts:
            raise ModelingError("moe_route_expert_token_counts_empty")
        row_values.append(sum(declared_counts))
    if not row_values and ids is not None:
        row_values.append(len(ids))
    if row_values and len(set(row_values)) != 1:
        raise ModelingError(
            f"moe_route_local_rows_conflict:{sorted(set(row_values))}")
    total = row_values[0] if row_values else None
    if total is None:
        return MoeRoute(
            basis=ROUTE_BASIS_UNRESOLVED, scope="local", rows=None,
            expert_range=local_range, source="router_received_rows_ledger",
            unresolved=tuple(dict.fromkeys(
                extra + ["router_received_rows_unresolved"])))

    # A full local count vector is a range fact; max(ids)+1 is not.
    range_candidates: list[tuple[int, str]] = []
    if local_range is not None:
        range_candidates.append((int(local_range), "active_expert_range"))
    if declared_counts is not None and sum(declared_counts) == total:
        range_candidates.append((len(declared_counts), "router_expert_token_counts"))
    if geometry_values is not None and sum(geometry_values) == total:
        range_candidates.append((len(geometry_values), geometry_source or
                                 "geometry_routing_counts"))
    if (local_range is not None and declared_counts is not None
            and len(declared_counts) != int(local_range)):
        raise ModelingError(
            "moe_route_expert_token_counts_scope_mismatch:"
            f"{len(declared_counts)}!={int(local_range)}")
    distinct_ranges = {value for value, _source in range_candidates}
    if len(distinct_ranges) > 1:
        raise ModelingError(
            f"moe_route_local_range_conflict:{sorted(distinct_ranges)}")
    if local_range is None and distinct_ranges:
        local_range = distinct_ranges.pop()
    if local_range is not None:
        if int(local_range) < 1 or int(local_range) > global_experts:
            raise ModelingError(
                "moe_route_local_range_out_of_bounds:"
                f"{int(local_range)}!={global_experts}")
        local_range = int(local_range)

    forward = inverse = assignment = assignment_tokens = assignment_slots = None
    identity_counts = None
    identity_source = ""
    identity_unresolved: list[str] = []
    if ids is not None:
        if len(ids) != total:
            identity_unresolved.append(
                "router_local_expert_ids_do_not_match_evaluated_rows")
        else:
            domain = int(local_range) if local_range is not None else global_experts
            (identity_counts, _offsets, forward, inverse, assignment,
             assignment_tokens, assignment_slots) = layout_from_ids(ids, domain)
            identity_source = "router_local_expert_ids"
            if local_range is None:
                # The assignment order is known, but the full range and the
                # complete count vector are not.  Do not promote max(ids)+1.
                identity_counts = None

    declared_complete = None
    if declared_counts is not None:
        if sum(declared_counts) != total:
            raise ModelingError(
                "moe_route_expert_token_counts_rows_mismatch:"
                f"{sum(declared_counts)}!={total}")
        if local_range is not None and len(declared_counts) != int(local_range):
            raise ModelingError(
                "moe_route_expert_token_counts_scope_mismatch:"
                f"{len(declared_counts)}!={int(local_range)}")
        declared_complete = declared_counts

    geometry_complete = None
    if geometry_values is not None:
        if sum(geometry_values) != total:
            extra.append("geometry_routing_counts_rows_mismatch")
        elif local_range is not None and len(geometry_values) != int(local_range):
            extra.append("geometry_routing_counts_scope_mismatch")
        elif local_range is None and len(geometry_values) == 0:
            extra.append("geometry_routing_counts_empty")
        else:
            geometry_complete = geometry_values

    if identity_counts is not None and declared_complete is not None \
            and identity_counts != declared_complete:
        raise ModelingError("moe_route_ids_counts_conflict")
    selected_counts = (identity_counts if identity_counts is not None
                       else declared_complete)
    if geometry_complete is not None and selected_counts is not None \
            and tuple(geometry_complete) != tuple(selected_counts):
        extra.append("geometry_route_disagrees_with_selected_route")
    counts = None
    basis = ROUTE_BASIS_UNRESOLVED
    source = ""
    if identity_counts is not None:
        counts = identity_counts
        basis = ROUTE_BASIS_IDS
        source = identity_source
    elif declared_complete is not None:
        counts = declared_complete
        basis = ROUTE_BASIS_COUNTS
        source = "router_expert_token_counts"
    elif geometry_complete is not None:
        counts = geometry_complete
        basis = ROUTE_BASIS_GEOMETRY
        source = geometry_source or "geometry_routing_counts"
        if local_range is None:
            local_range = len(geometry_complete)

    if counts is not None:
        if local_range is not None and len(counts) != int(local_range):
            raise ModelingError(
                "moe_route_expert_token_counts_scope_mismatch:"
                f"{len(counts)}!={int(local_range)}")
        if sum(counts) != total:
            raise ModelingError(
                f"moe_route_rows_mismatch:{sum(counts)}!={total}")

    if control == "balanced":
        if counts is not None:
            raise ModelingError(
                "moe_route_actual_route_and_balanced_control_conflict")
        if local_range is None:
            raise ModelingError(
                "moe_route_balanced_control_requires_expert_range")
        counts = balanced_counts(total, int(local_range))
        basis = ROUTE_BASIS_BALANCED
        source = "router_route_control=balanced"
        extra.append("forced_balanced_token_identities_not_declared")

    if counts is None:
        if forward is not None:
            return make_route(
                basis=ROUTE_BASIS_IDS, scope="local", counts=None,
                rows=total, source=identity_source, forward=forward,
                inverse=inverse, assignment_experts=assignment,
                assignment_tokens=assignment_tokens,
                assignment_slots=assignment_slots,
                identity_space=("row_index" if forward is not None
                                 else "not_declared"),
                expert_range=local_range, counts_complete=False,
                unresolved=tuple(extra + identity_unresolved
                                 + ["expert_token_counts_unresolved"]))
        return MoeRoute(
            basis=ROUTE_BASIS_UNRESOLVED, scope="local", rows=total,
            expert_range=local_range, source="no_complete_local_route",
            unresolved=tuple(dict.fromkeys(
                extra + identity_unresolved
                + ["expert_token_counts_unresolved"])))
    return make_route(
        basis=basis, scope="local", counts=counts, rows=total,
        source=source, forward=forward, inverse=inverse,
        assignment_experts=assignment,
        assignment_tokens=assignment_tokens, assignment_slots=assignment_slots,
        identity_space=("row_index" if forward is not None
                        else "not_declared"),
        expert_range=local_range, counts_complete=True,
        performance_control=(basis == ROUTE_BASIS_BALANCED),
        unresolved=tuple(extra + identity_unresolved))


def work_record(*, issued: Any, useful: Any = None, source: str = "") -> dict:
    """Normalized work with one unit per resource; useful work is not charged."""
    def number(value: Any) -> int | float:
        amount = float(value)
        return int(amount) if amount.is_integer() else amount

    issued_values = {str(key): number(value)
                     for key, value in dict(issued).items() if float(value) > 0}
    useful_values = {str(key): number(value)
                     for key, value in dict(useful or {}).items()
                     if float(value) > 0}
    units = {key: WORK_UNITS.get(key, "work") for key in issued_values}
    for key in useful_values:
        units.setdefault(key, WORK_UNITS.get(key, "work"))
    return {
        "mte2_bytes": issued_values.get("mte2", 0.0),
        "mte3_bytes": issued_values.get("mte3", 0.0),
        "hbm_bytes": issued_values.get("hbm", 0.0),
        "vector_ops": issued_values.get("vector", 0.0),
        "resources": issued_values,
        "useful_resources": useful_values,
        "units": units,
        "source": source,
    }


@dataclass(frozen=True)
class NativeMoeProgram(KernelProgram):
    """One KernelProgram plus its typed call fragment projection.

    This is not a second executor: ``start`` is the existing KernelProgram
    path.  The typed fragment supplies attrs/ports and the aggregate work
    summary; ``stages`` is the only charged program.
    """

    fragment: Any = None
    stage_records: tuple = ()
    call_id: str = ""
    phase: str = "prefill"
    output_stages: Any = None

    @property
    def attrs(self):
        return () if self.fragment is None else self.fragment.attrs

    @property
    def ports(self):
        return None if self.fragment is None else self.fragment.ports

    @property
    def occurrence(self):
        return None if self.fragment is None else self.fragment.occurrence

    @property
    def work(self):
        return None if self.fragment is None else self.fragment.work

    @property
    def service_work(self):
        return {} if self.fragment is None else self.fragment.service_work

    @property
    def name(self):
        return "" if self.fragment is None else self.fragment.name

    def submit_staged(self, conductor: Any, *, ready: Any, inputs: Any = None,
                      owner: str = "") -> dict:
        """Submit this same program through the caller's conductor bindings."""
        plan = {
            "call_id": self.call_id or self.name,
            "phase": self.phase,
            "fragment": self.fragment,
            "stages": tuple(self.stage_records),
            "output_stages": dict(self.output_stages or {}),
            "execution_program": self,
        }
        return execute_staged(conductor, plan, ready=ready, inputs=inputs,
                              owner=owner)

    def start_events(self, engine: Any, name: str, ready: Any, *,
                     owner: str = "") -> tuple[Any, dict[str, Any]]:
        """Start the same Stage DAG and retain its actual per-stage Events."""
        events: dict[str, Any] = {}

        def scoped(items):
            return scope_allocations(name, items)

        for stage in self.stages:
            predecessors = tuple(events[key] for key in stage.after) or (ready,)
            events[stage.name] = stage.submit(
                engine, f"{name}:{stage.name}", predecessors, scoped, owner)
        complete = engine.submit(
            name + ":complete", after=tuple(events.values()) or (ready,),
            owner=owner)
        return complete, events

    def to_dict(self) -> dict:
        stages = []
        for record in self.stage_records:
            row = dict(record)
            row["acquire"] = [item.to_dict() for item in record["acquire"]]
            row["free"] = [item.to_dict() for item in record["free"]]
            stages.append(row)
        return {
            "schema": "mindie-sim/native-moe-program/1",
            "fragment": (None if self.fragment is None
                         else self.fragment.to_dict()),
            "stages": stages,
            "call_id": self.call_id,
            "phase": self.phase,
            "output_stages": dict(self.output_stages or {}),
            "work": dict(self.service_work),
        }


def stage_record(name: str, work: Any, *, after: tuple[str, ...] = (),
                 coupled: bool = False, acquire: tuple = (), free: tuple = (),
                 tail: bool = False) -> dict:
    """One real Stage declaration; units come from the common work table."""
    amounts = {str(key): float(value) for key, value in dict(work).items()
               if float(value) > 0}
    return {
        "name": str(name), "work": amounts,
        "work_units": {key: WORK_UNITS.get(key, "work") for key in amounts},
        "after": tuple(after), "coupled": bool(coupled),
        "acquire": tuple(acquire), "free": tuple(free), "tail": bool(tail),
    }


def stage_totals(records: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in records:
        for resource, amount in record["work"].items():
            totals[resource] = totals.get(resource, 0.0) + float(amount)
    return totals


def kernel_program(records: Any):
    """Existing Stage/KernelProgram projection of the same stage records."""
    from mindie_sim.prediction.engine import Allocation
    from mindie_sim.prediction.program import KernelProgram, Stage

    def allocations(items):
        return tuple(Allocation(item.name, item.pool, item.size, item.units)
                     for item in items)

    return KernelProgram(tuple(
        Stage(name=record["name"], work=record["work"],
              after=tuple(record["after"]), coupled=bool(record["coupled"]),
              acquire=allocations(record["acquire"]),
              free=allocations(record["free"]),
              work_units=record["work_units"])
        for record in records))


def native_program(plan: dict) -> NativeMoeProgram:
    """Project one plan's typed fragment and charged Stage program."""
    program = plan.get("execution_program")
    if program is None:
        raise ModelingError("moe_plan_execution_program_missing")
    return NativeMoeProgram(
        stages=tuple(program.stages), minimum_ms=float(program.minimum_ms),
        fragment=plan["fragment"], stage_records=tuple(plan.get("stages", ())),
        call_id=str(plan.get("call_id") or ""),
        phase=str(plan.get("phase") or "prefill"),
        output_stages=dict(plan.get("output_stages") or {}))


def _row_tiles(count: int, capacity: int | None,
               row_bytes: int) -> tuple[tuple[int, int], ...]:
    """Split a real row count into capacity-bounded tiles, never padding."""
    count, row_bytes = int(count), int(row_bytes)
    if count < 1 or row_bytes < 1:
        return ()
    if capacity is None:
        return ((0, count),)
    capacity = int(capacity)
    if capacity < row_bytes:
        raise ModelingError(
            f"moe_row_tile_exceeds_workspace:{row_bytes}!={capacity}")
    rows = max(1, capacity // row_bytes)
    return tuple((start, min(count, start + rows))
                 for start in range(0, count, rows))


def _stage_fragment(record: dict, *, call_id: str, phase: str):
    """One charged Stage as a typed conductor fragment (no free here)."""
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        ProgramFragment, WorkItem, WorkSet)
    items = tuple(
        WorkItem(resource, amount, record["work_units"].get(resource, "work"),
                 kind="issued", source="native_moe_stage")
        for resource, amount in record["work"].items())
    return ProgramFragment(
        name=f"moe.stage.{record['name']}",
        occurrence=Occurrence("stage", record["name"], phase=phase,
                              invocation=call_id),
        work=WorkSet(items), allocations=tuple(record["acquire"]),
        coupled=bool(record["coupled"]), notes=("internal_stage",))


def execute_staged(conductor: Any, plan: dict, *, ready: Any,
                   inputs: Any = None, owner: str = "") -> dict:
    """Submit input admission, charged Stages and the typed call join.

    Every charged Stage goes through the supplied conductor, so its
    ``resource_bindings``, ``pool_bindings``, unit conversions, memory ledger,
    owner and event registry are the same ones used by admission/join.  Block
    allocations are acquired on the read stage and explicitly held until the
    block's write completion; no Stage is sent to the naked Engine.
    """
    if plan.get("execution_program") is None or not plan.get("stages"):
        raise ModelingError("moe_plan_execution_program_missing")
    from dataclasses import replace
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import ProgramFragment, WorkSet
    from mindie_sim.schema.subgraphs import PortSet
    fragment = plan["fragment"]
    bound_inputs = dict(inputs or {})
    phase = fragment.occurrence.phase if fragment.occurrence else "prefill"
    engine = conductor.engine
    # Admission is a zero-work typed gate: it validates and waits on the real
    # producer/output events before the charged Stage program is submitted.
    gate = ProgramFragment(
        name=f"{fragment.name}.admission",
        occurrence=Occurrence("stage", "input_admission", phase=phase,
                              invocation=plan["call_id"]),
        ports=PortSet(tensor_in=fragment.ports.tensor_in),
        work=WorkSet(), notes=("input_admission",))
    admission = conductor.submit(
        gate, after=(ready, *tuple(bound_inputs.values())),
        inputs=bound_inputs, owner=owner)

    holds: dict[str, Any] = {}
    for record in plan["stages"]:
        for effect in record["acquire"]:
            holds.setdefault(
                effect.name,
                engine.future(f"{plan['call_id']}.last_use.{effect.name}"))
    stage_events: dict[str, Any] = {}
    stage_handles = []
    release_handles = []
    grants: dict[str, Any] = {}
    for record in plan["stages"]:
        predecessors = tuple(stage_events[key] for key in record["after"]) \
            or (admission.event,)
        stage_holds = {
            effect.name: holds[effect.name] for effect in record["acquire"]}
        handle = conductor.submit(
            _stage_fragment(record, call_id=plan["call_id"], phase=phase),
            after=predecessors, holds=stage_holds, owner=owner)
        stage_handles.append(handle)
        stage_events[record["name"]] = handle.event
        for name, event in handle.grant_by_name:
            grants[name] = event
        # The held allocation is released only after the stage that last uses
        # the block has completed.  This is a zero-work release boundary, not a
        # charged Stage; T08 may bind the same future to its memory ledger.
        for effect in record["free"]:
            hold = holds[effect.name]

            def release_last_use(_event, hold=hold):
                engine.resolve(hold)

            release_handles.append(engine.submit(
                f"moe.release.{plan['call_id']}.{effect.name}",
                after=(handle.event,), owner=owner,
                on_complete=release_last_use))

    output_stages = dict(plan.get("output_stages") or {})
    known_ports = []
    bindings: dict[str, Any] = {}
    prefix = f"{plan['call_id']}."
    for port in fragment.ports.event_out:
        base = port.name[len(prefix):-len(".complete")]
        stage_name = output_stages.get(base)
        if stage_name is not None and stage_name in stage_events:
            known_ports.append(port)
            bindings[port.name] = stage_events[stage_name]
    join_ports = replace(fragment.ports, event_out=tuple(known_ports))
    join = replace(fragment, work=WorkSet(), ports=join_ports)
    call_handle = conductor.submit(
        join, after=tuple(stage_events.values()) or (admission.event,),
        inputs=bound_inputs, event_bindings=bindings, owner=owner)
    outputs = {
        port.name: conductor.event(port.name, port.generation)
        for port in known_ports
    }
    return {
        "completion": call_handle.event,
        "stage_completion": call_handle.event,
        "stage_events": stage_events,
        "stage_handles": tuple(stage_handles),
        "release_handles": tuple(release_handles),
        "grants": dict(grants),
        "call_handle": call_handle,
        "admission": admission.event,
        "outputs": outputs,
    }


def build_fragment(*, name: str, call_id: str, phase: str,
                   inputs: tuple[tuple[str, str, str], ...],
                   outputs: tuple[tuple[str, str, str], ...],
                   work: dict | None, attrs: tuple[tuple[str, Any, str, str], ...],
                   notes: tuple[str, ...],
                   event_outputs: tuple[str, ...] | None = None):
    """Build the existing common fragment; no new executor or timer is created."""
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        ProgramFragment, SemanticAttr, WorkItem, WorkSet)
    from mindie_sim.schema.subgraphs import (
        EventPortSpec, PortSet, TensorPortSpec)
    ports = PortSet(
        tensor_in=tuple(TensorPortSpec(port, "in", representation, dtype=dtype)
                        for port, dtype, representation in inputs),
        tensor_out=tuple(TensorPortSpec(port, "out", representation, dtype=dtype)
                         for port, dtype, representation in outputs),
        event_out=tuple(
            EventPortSpec(f"{call_id}.{port}.complete", "out", "complete",
                          scope="operator", owner="self")
            for port in (tuple(item[0] for item in outputs)
                         if event_outputs is None else event_outputs)),
    )
    work = work or {"resources": {}, "useful_resources": {},
                    "units": {}, "source": "typed_call_join"}
    items = [
        WorkItem(resource, amount, work["units"].get(resource, "work"),
                 kind="issued", source=work.get("source", ""))
        for resource, amount in work["resources"].items()
    ]
    items.extend(
        WorkItem(resource, amount, work["units"].get(resource, "work"),
                 kind="useful", source=work.get("source", ""))
        for resource, amount in work["useful_resources"].items()
    )
    return ProgramFragment(
        name=name,
        occurrence=Occurrence("call", call_id, phase=phase or "prefill",
                              invocation=call_id),
        ports=ports,
        work=WorkSet(tuple(items)),
        attrs=tuple(SemanticAttr(attr, value, unit=unit, source=source)
                    for attr, value, unit, source in attrs),
        notes=tuple(notes),
    )


def program_record(*, common_program: str, stage: str, work: dict,
                   fragment: Any, route: MoeRoute, decomposition: dict,
                   fusion_boundary: dict, alignment: dict | None = None,
                   stage_records: Any = None) -> dict:
    """Serializable view of the one-stage common program for this native call."""
    if stage_records is not None:
        stages = []
        for record in stage_records:
            row = dict(record)
            row["acquire"] = [item.to_dict() for item in record["acquire"]]
            row["free"] = [item.to_dict() for item in record["free"]]
            stages.append(row)
    else:
        stages = [{"name": stage, "after": [], "work": dict(work["resources"]),
                   "work_units": dict(work["units"]), "coupled": False,
                   "tail": False, "acquire": [], "free": []}]
    payload = {
        "schema": "mindie-sim/moe-native-program/1",
        "common_program": common_program,
        "stages": stages,
        "stage_count": len(stages),
        "work": dict(work["resources"]),
        "useful_work": dict(work["useful_resources"]),
        "work_source": work.get("source", ""),
        "output_events": [port.name for port in fragment.ports.event_out],
        "route_basis": route.basis,
        "route_layout_digest": route.layout_digest,
        "decomposition": dict(decomposition),
        "fusion_boundary": dict(fusion_boundary),
        "fragment": fragment.to_dict(),
    }
    if alignment is not None:
        payload["alignment"] = dict(alignment)
    return payload



def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if source_revision_from_oci(image, image_id) != "r64":
        return {}
    return meta


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 outgoing TokenPermute call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Decode, non-R64 images, another routing
    row and the separate ``npu_moe_init_routing_v2`` capability return
    ``None`` so ordinary prediction is unchanged.
    """
    if str(getattr(compiled, "kind", "") or "") != "moe_router":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "router"
            or str(getattr(operator, "name", "") or "") != "router_permute"):
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "")
                     or meta.get("phase") or "")
    if phase_name != "prefill":
        return None
    theta = operator.theta
    operation = theta.find("router_source_operation", None)
    if (not isinstance(operation, str)
            or not operation.startswith("MoeTokenPermute")):
        return None
    selected = str(theta.find("router_selected", SOURCE_SELECTED) or "")
    if selected not in {"", SOURCE_SELECTED}:
        return None
    declared_dtype = str(theta.find("router_source_dtype", "bf16") or "bf16")
    dtype_bytes = _int(theta.find("router_dtype_bytes", 2), 2)
    indices_bytes = _int(theta.find("router_indices_bytes", 4), 4)
    hidden = _int(theta.find("router_hidden", 0))
    top_k = _int(theta.find("k", 0))
    experts = _int(theta.find("E", 0))
    if (declared_dtype != "bf16" or dtype_bytes != 2
            or indices_bytes != 4 or hidden < 1 or top_k < 1):
        return None
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_source_status="kernel_binary_not_retained")
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "prefill",
        # `sorted_indices` is a 1-D M*k row vector with no occurrence-
        # independent suffix; the dynamic occurrence carries its exact shape
        # while the other roles keep their static invariant suffixes.
        "shapes": {
            "tokens": [hidden],
            "indices": [top_k],
            "permuted_tokens": [hidden],
        },
        "attrs": {
            "input_contract": ["tokens", "indices", "num_out_tokens"],
            "output_contract": ["permuted_tokens", "sorted_indices"],
            "router_hidden": hidden,
            "hidden": hidden,
            "k": top_k,
            "E": experts,
            "router_dtype_bytes": dtype_bytes,
            "router_indices_bytes": indices_bytes,
            "router_source_operation": operation,
            "router_source_dtype": declared_dtype,
            "router_source_device": "AIV",
            "router_selected": selected or SOURCE_SELECTED,
            "token_permute_role": ROLE,
            "deployment_image_id": str(meta.get("image_id") or
                                       meta.get("deployment_image_id") or
                                       R64_IMAGE_SHA),
            "deployment_source_revision": "r64",
            "source_commit": R64_SOURCE_COMMIT,
            "source_lines": list(SOURCE_LINES),
            "source_num_out_tokens_line": NUM_OUT_TOKENS_LINE,
            "acl_api": ACLNN_API,
            "op_type": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "kernel_source_status": "kernel_binary_not_retained",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "num_out_tokens_source": "topk_ids.numel()",
            "num_out_tokens_expr": "M*k",
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=("aclnn_symbol_unresolved",
                    "moe_token_permute_kernel_source_not_retained",
                    "moe_token_permute_binary_selection_unresolved"))


def _permute_route(compiled: Any, geometry: Any, attrs: Any, *, tokens: int,
                   top_k: int, experts: int) -> MoeRoute:
    """One normalized route fact for this evaluated outbound call."""
    if experts < 1:
        return MoeRoute(
            basis=ROUTE_BASIS_UNRESOLVED, scope="global",
            rows=int(tokens) * int(top_k), expert_range=None,
            source="binding_missing_expert_count",
            unresolved=("expert_count_unresolved",))
    declared = route_inputs(compiled, attrs)
    geometry_source = ""
    geometry_counts = None
    if hasattr(geometry, "routing_for"):
        layer = int(getattr(compiled, "layer", 0) or 0)
        geometry_counts = geometry.routing_for(layer)
        geometry_source = f"geometry.routing[layer={layer}]"
    return normalize_route(
        declared=declared, geometry_counts=geometry_counts, scope="global",
        experts=int(experts), rows=int(tokens) * int(top_k),
        identity="topk_ids", tokens=int(tokens), top_k=int(top_k),
        geometry_source=geometry_source)


def _permute_fragment(call_id: str, route: MoeRoute, *, tokens: int,
                      top_k: int, hidden: int, rows: int, work: dict):
    attrs = (
        ("native_api", ENTRY, "", "declared_source"),
        ("acl_api", ACLNN_API, "", "declared_source"),
        ("route_basis", route.basis, "", "occurrence"),
        ("route_layout_digest", route.layout_digest, "", "occurrence"),
        ("sorted_indices_direction", "inverse_assignment", "", "ABI"),
        ("assignment_identity", route.to_dict()["assignment_identity"], "",
         "occurrence"),
        ("expert_token_counts", route.counts or (), "rows", "occurrence"),
        ("expert_token_offsets", route.offsets or (), "rows", "occurrence"),
        ("active_experts", route.active_experts, "experts", "occurrence"),
        ("m", int(tokens), "rows", "geometry"),
        ("k", int(top_k), "experts_per_token", "binding"),
        ("hidden", int(hidden), "elements", "binding"),
        ("num_out_tokens", int(rows), "rows",
         "source_expression_topk_ids_numel"),
    )
    return build_fragment(
        name="moe.token_permute", call_id=call_id, phase="prefill",
        inputs=(("tokens", "bf16", "dense_activation"),
                ("indices", "int32", "expert_identity")),
        outputs=(("permuted_tokens", "bf16", "expert_grouped_activation"),
                 ("sorted_indices", "int32",
                  "inverse_flattened_assignment_index")),
        work=work, attrs=attrs,
        notes=("single_api_call", "no_transport_bytes_charged",
               "no_grouped_expert_compute_charged"))


def _permute_program(route: MoeRoute, work: dict, fragment: Any,
                     *, tokens: int, top_k: int, hidden: int,
                     rows: int, stage_records: Any = None) -> dict:
    dense = float(int(tokens) * int(hidden) * 2)
    indices = float(int(rows) * 4)
    packed = float(int(rows) * int(hidden) * 2)
    return program_record(
        common_program="moe.F1.router_permute",
        stage="permute_rows_by_expert",
        work=work, fragment=fragment, route=route,
        stage_records=stage_records,
        decomposition={
            "api_calls": 1, "kernels": [KERNEL_NAME],
            "members": ["router_permute"],
            "separate_from": ["dispatch_alltoall", "grouped_expert_matmul"],
            "not_charged": ["collective transport", "expert compute"],
        },
        fusion_boundary={
            "owner": "F1:router:router_permute",
            "charge": "one_per_native_call",
            "standalone_prefill_only": True,
            "fused_mc2_contains_dispatch": True,
        },
        alignment={
            "family_template": "router_permute",
            "family_read_bytes": dense + indices,
            "native_read_bytes": work["resources"].get("mte2", 0.0),
            "family_write_bytes": packed + indices,
            "native_write_bytes": work["resources"].get("mte3", 0.0),
            "family_vector_basis": "indices_bytes",
            "native_vector_basis": "one_row_address_operation_per_assignment",
            "note": ("typed read/write bytes agree with the shared router "
                     "template; the vector unit is stated explicitly"),
        })


def _permute_materialization(tokens: int, top_k: int, hidden: int) -> dict:
    rows = int(tokens) * int(top_k)
    dense = int(tokens) * int(hidden) * 2
    indices = rows * 4
    packed = rows * int(hidden) * 2
    return {
        "inputs": {"tokens": dense, "indices": indices},
        "outputs": {"permuted_tokens": packed, "sorted_indices": indices},
        "input_read_bytes": dense + indices,
        "output_write_bytes": packed + indices,
        "total_bytes": dense + indices + packed + indices,
        "num_out_tokens": rows,
    }


def _permute_stages(route: MoeRoute, *, tokens: int, top_k: int,
                    hidden: int, block_slots: int | None) -> list[dict]:
    """Read, per-block sort and write stages with optional output slots."""
    from mindie_sim.schema.operators import AllocationEffect
    rows = int(tokens) * int(top_k)
    dense = int(tokens) * int(hidden) * 2
    indices = rows * 4
    sort_ops = int(top_k)
    sort_factor = max(1, (sort_ops - 1).bit_length())
    records = [
        stage_record("read_tokens", {"mte2": dense, "hbm": dense},
                     coupled=True),
        stage_record("read_indices", {"mte2": indices, "hbm": indices},
                     coupled=True),
    ]
    pool = "moe_permute_block"

    def grant(label: str, size: int):
        if block_slots is None:
            return (), ()
        if size > int(block_slots):
            raise ModelingError(
                f"moe_plan_permute_block_exceeds_workspace:{size}!={block_slots}")
        name = f"block_slot_{label}"
        return (
            (AllocationEffect.acquire(name, pool, size),),
            (AllocationEffect.release(name, pool, size),),
        )

    if route.counts is not None and route.counts_complete:
        groups = [(index, value) for index, value in enumerate(route.counts)
                  if value > 0]
    elif route.observed_counts:
        groups = list(route.observed_counts)
    else:
        groups = [(-1, rows)]
    write_names: list[str] = []
    row_out = int(hidden) * 2 + 4
    for position, (expert, count) in enumerate(groups):
        base = "all" if expert < 0 else f"e{expert}"
        tiles = _row_tiles(count, block_slots, row_out)
        for tile_index, (start, end) in enumerate(tiles):
            tile_count = end - start
            label = (base if len(tiles) == 1
                     else f"{base}t{tile_index}")
            sort = f"sort_{label}"
            write = f"write_{label}"
            block_bytes = tile_count * row_out
            acquire, release = grant(label, block_bytes)
            records.append(stage_record(
                sort, {"vector": tile_count * sort_factor},
                after=("read_tokens", "read_indices"), acquire=acquire))
            records.append(stage_record(
                write, {"mte3": block_bytes,
                        "hbm": block_bytes},
                after=(sort,), free=release,
                tail=(position == len(groups) - 1
                      and tile_index == len(tiles) - 1), coupled=True))
            write_names.append(write)
    if write_names:
        records.append(stage_record("publish_outputs", {},
                                    after=tuple(write_names)))
    return records


def plan_permute(*, call_id: str, phase: str, tokens: int, top_k: int,
                 hidden: int, route: MoeRoute,
                 block_slots: int | None = None) -> dict:
    """Single leaf supply for one normalized outbound TokenPermute occurrence."""
    tokens, top_k, hidden = int(tokens), int(top_k), int(hidden)
    rows = tokens * top_k
    if route.rows is not None and int(route.rows) != rows:
        raise ModelingError(
            f"moe_plan_permute_route_rows_mismatch:{route.rows}!={rows}")
    if route.experts is not None and int(route.experts) <= 0:
        raise ModelingError("moe_plan_permute_route_experts_invalid")
    actual_shapes = {
        "tokens": [tokens, hidden],
        "indices": [tokens, top_k],
        "permuted_tokens": [rows, hidden],
        "sorted_indices": [rows],
    }
    descriptors = {
        "tokens": _descriptor([tokens, hidden], "bf16", "ND", "input"),
        "indices": _descriptor([tokens, top_k], "int32", "ND", "input"),
        "permuted_tokens": _descriptor([rows, hidden], "bf16", "ND",
                                       "output"),
        "sorted_indices": _descriptor([rows], "int32", "ND", "output"),
        "num_out_tokens": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": rows, "pointer": None,
            "allocation": "caller",
        },
    }
    arguments = {
        "tokens": "tokens", "indices": "indices", "num_out_tokens": rows,
    }
    materialization = _permute_materialization(tokens, top_k, hidden)
    stages = _permute_stages(
        route, tokens=tokens, top_k=top_k, hidden=hidden,
        block_slots=block_slots)
    work = work_record(
        issued=stage_totals(stages),
        useful={"hbm": float(materialization["inputs"]["tokens"]
                             + materialization["outputs"]["permuted_tokens"])},
        source="native_moe_stage_program")
    fragment = _permute_fragment(
        call_id, route, tokens=tokens, top_k=top_k, hidden=hidden, rows=rows,
        work=work)
    logical = {
        "dtype": "bf16", "layout": "ND", "shapes": actual_shapes,
        "tokens_dtype": "bf16", "indices_dtype": "int32",
        "sorted_indices_dtype": "int32",
    }
    program = _permute_program(
        route, work, fragment, tokens=tokens, top_k=top_k, hidden=hidden,
        rows=rows, stage_records=stages)
    allocation = {
        "inputs": "external_owner", "outputs": "external_owner",
        "internal": [], "workspace": "not_declared",
        "lifetimes": "owner_retained",
    }
    if block_slots is not None:
        allocation["internal"] = [{
            "name": "moe_permute_block", "pool": "moe_permute_block",
            "capacity_bytes": int(block_slots), "lifetime": "per_block"}]
        allocation["workspace"] = {
            "kind": "token_block", "pool": "moe_permute_block",
            "capacity_bytes": int(block_slots), "last_use": "block_write"}
    output_stages = {
        "permuted_tokens": "publish_outputs",
        "sorted_indices": "publish_outputs",
    }
    unrecognized = (
        "sort_algorithm_selection_not_retained",
        "source_row_gather_locality_not_modeled",
        "tile_assignment_boundary_not_retained",
    )
    program.update({
        "logical": logical, "materialization": materialization,
        "issued_work": work["resources"], "route": route.to_dict(),
        "allocation": allocation,
        "output_stages": dict(output_stages),
        "unrecognized_work": list(unrecognized),
    })
    return {
        "schema": "mindie-sim/moe-native-leaf-plan/1",
        "call": ENTRY,
        "call_id": str(call_id),
        "phase": str(phase),
        "route": route,
        "logical": logical,
        "descriptors": descriptors,
        "arguments": arguments,
        "materialization": materialization,
        "allocation": allocation,
        "unrecognized_work": unrecognized,
        "stages": tuple(stages),
        "execution_program": kernel_program(stages),
        "output_stages": output_stages,
        "work": work,
        "fragment": fragment,
        "program": program,
    }


def program_fragment(binding: Any, compiled: Any, geometry: Any):
    """Independent typed fragment for the selected outbound TokenPermute."""
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("token_permute_role") != ROLE
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("op_type") != KERNEL_NAME
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if not str(attrs.get("router_source_operation") or "").startswith(
            "MoeTokenPermute"):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    hidden = _int(attrs.get("router_hidden") or attrs.get("hidden"))
    top_k = _int(attrs.get("k"))
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if hidden < 1 or top_k < 1 or tokens < 1:
        return None
    route = _permute_route(compiled, geometry, attrs, tokens=tokens,
                           top_k=top_k, experts=_int(attrs.get("E")))
    plan = plan_permute(
        call_id=binding.call_id, phase="prefill", tokens=tokens,
        top_k=top_k, hidden=hidden, route=route)
    if plan.get("fragment") is None or plan.get("execution_program") is None:
        return None
    return native_program(plan)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected outbound TokenPermute.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is the
    ``router_permute`` call itself; the caller's dispatch/collective work is
    not charged here.
    """
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("token_permute_role") != ROLE
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("op_type") != KERNEL_NAME
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if not str(attrs.get("router_source_operation") or "").startswith(
            "MoeTokenPermute"):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    hidden = _int(attrs.get("router_hidden") or attrs.get("hidden"))
    top_k = _int(attrs.get("k"))
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if hidden < 1 or top_k < 1 or tokens < 1:
        return None
    route = _permute_route(compiled, geometry, attrs, tokens=tokens,
                           top_k=top_k, experts=_int(attrs.get("E")))
    plan = plan_permute(
        call_id=binding.call_id, phase="prefill", tokens=tokens,
        top_k=top_k, hidden=hidden, route=route)
    rows = tokens * top_k
    cache_key = (binding.call_id, tokens, hidden, top_k, route.basis,
                 route.performance_control, route.source, route.unresolved,
                 route.layout_digest)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = plan["logical"]["shapes"]
    actual_attrs = {
        **attrs,
        "m": tokens,
        "num_out_tokens": rows,
        "padded_mode": False,
        "route_basis": route.basis,
        "expert_token_counts": (list(route.counts)
                                if route.counts is not None else None),
        "expert_token_offsets": (list(route.offsets)
                                 if route.offsets is not None else None),
    }
    actual = {"dtype": "bf16", "layout": "ND", "shapes": actual_shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "source_status": "kernel_binary_not_retained"})
    prepare = {
        "schema": "mindie-sim/moe-token-permute-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R64_SOURCE_COMMIT,
        "source_lines": list(SOURCE_LINES),
        "source_num_out_tokens_line": NUM_OUT_TOKENS_LINE,
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": plan["arguments"],
        "argument_order": ["tokens", "indices", "num_out_tokens"],
        "shapes": actual_shapes,
        "descriptors": plan["descriptors"],
        "arrays": plan["descriptors"],
        "logical": plan["logical"],
        "materialization": plan["materialization"],
        "optional_slots": {
            "num_out_tokens": "host_scalar_from_topk_ids_numel",
            "padded_mode": "not_passed_default_false",
        },
        "match": {
            "native_type": "MoeTokenPermute",
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "m": tokens,
            "k": top_k,
            "hidden": hidden,
            "num_out_tokens": rows,
            "source_sha256": source.get("sha256"),
            "source_lines": list(SOURCE_LINES),
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "source_commit": R64_SOURCE_COMMIT,
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "revision": REVISION,
            "selection": ("exact R64 image digest + caller blob invariant "
                          "over the R64 device_op.py blob"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "device_op_sha256": R64_DEVICE_OP_SHA,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_nn_kernel_binary_not_retained",
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
        },
        "route": route.to_dict(),
        "expert_token_counts": (list(route.counts)
                                if route.counts is not None else None),
        "expert_token_offsets": (list(route.offsets)
                                 if route.offsets is not None else None),
        "sorted_indices": (list(route.inverse)
                           if route.inverse is not None else None),
        "assignment": {
            "identity_space": route.identity_space,
            "forward": (list(route.forward)
                        if route.forward is not None else None),
            "inverse": (list(route.inverse)
                        if route.inverse is not None else None),
            "tokens": (list(route.assignment_tokens)
                       if route.assignment_tokens is not None else None),
            "slots": (list(route.assignment_slots)
                      if route.assignment_slots is not None else None),
        },
        "work": plan["work"],
        "issued_work": plan["work"]["resources"],
        "program": plan["program"],
        "quant_metadata": {
            "tokens_dtype": "bf16",
            "tokens_bytes": 2,
            "indices_dtype": "int32",
            "indices_bytes": 4,
            "permuted_dtype": "bf16",
            "permuted_bytes": 2,
            "sorted_indices_dtype": "int32",
            "num_out_tokens_dtype": "int64_host_scalar",
            "num_out_tokens_expression": "M*k",
            "source": "selected R64 BF16 dense activation and INT32 ids",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": ["moe_token_permute_kernel_binary_not_retained",
                            "aclnn_symbol_unresolved",
                            *(["expert_token_counts_unresolved"]
                              if route.counts is None else [])],
        "unresolved": ["aclnn_symbol_unresolved",
                       "moe_token_permute_kernel_source_not_retained",
                       "moe_token_permute_binary_selection_unresolved",
                       "execution_rank_not_selected",
                       *list(route.unresolved)],
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration", "no_kernel_source_claim"],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "bf16", "layout": "ND",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
