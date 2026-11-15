"""A5-style explicit MoE token routes through the shared routing/transport glue.

The route is one integer source-token -> logical-expert ledger with a physical
expert placement.  These tests establish the behaviour the ordinary builders
must preserve when that ledger replaces a balanced performance control:

* every physical source rank owns its own source rows while the expert bank is
  a destination projection;
* dispatch and return traffic keep the exact integer peer matrix and the
  actual forward/return row widths (including quantized row scales);
* grouped work and group-list rows rebind to this route, not to a cached
  balanced histogram;
* a declared balanced scenario remains a separate explicit control.
"""
from dataclasses import asdict, replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.ops import comm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.direct_peer import DirectPeerCollective
from mindie_sim.prediction.moe_callers import (
    LOCAL_EXPERT_REDUCE, local_expert_reduce_recipe)
from mindie_sim.prediction.operator_analysis import cohort_operator_costs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.routing_realizations import select_realization
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView
from test_prediction_system import case


TOKENS = 32
TOP_K = 2
EXPERTS = 8
EP = 4
BANK = EXPERTS // EP
# Four deterministic source-local patterns; repeated with a source offset so
# every rank has a different but valid top-k row.
ROWS = ((0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7))


def _ids(owner_counts):
    ids, owners = [], []
    for source, count in enumerate(owner_counts):
        for index in range(count):
            ids.append(ROWS[(index + source) % len(ROWS)])
            owners.append(source)
    return ids, owners


def _placement():
    return tuple((expert // BANK, expert % BANK) for expert in range(EXPERTS))


def _case(*, tokens=TOKENS, quantized=False, route=None, balanced=False):
    raw = case(tp=4, ep=4, prompt=tokens, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=EXPERTS, top_k=TOP_K,
        hidden_size=32, moe_intermediate_size=32,
    )
    raw["machine"]["device"]["compute_peaks"].setdefault("int8", 1e9)
    raw["runtime"]["max_num_batched_tokens"] = max(16, tokens)
    if quantized:
        raw["model"]["quant"] = asdict(quant_preset("w8a8_int8"))
        raw["runtime"]["kernels"] = {"moe_transport": {
            "kind": "separate_quantized_payload", "payload_dtype": "int8",
            "scale_dtype": "fp32", "scale_granularity": "row",
            "histogram_dtype": "int64",
        }}
    if balanced:
        raw["runtime"]["routing"] = {"policy": "balanced"}
    elif route is not None:
        owner_counts, cohort = route
        ids, owners = _ids(owner_counts)
        item = {
            "scope": "body", "layer": 0, "phase": "prefill",
            "tokens": len(ids), "topk_ids": ids,
            "expert_placement": _placement(), "source_owners": owners,
        }
        if cohort:
            item["cohort"] = True
        raw["runtime"]["routing"] = {"realizations": [item]}
    return raw


def _ops(raw, *, tokens):
    return {op.op_id: op for op in target_operators(
        raw, phase="prefill", tokens=tokens, scope="body")}


def _realization(raw, *, tokens, rank):
    parsed = parse_case(raw)
    return select_realization(
        parsed.runtime, parsed.model.layer_plans[0], phase="prefill",
        tokens=tokens, valid_tokens=tokens, physical_rank=rank)


def _matrix(op):
    return dict(op.metadata)["peer_traffic"]["payload_bytes"]


def _off_diagonal(cells):
    return sum(value for source, row in enumerate(cells)
               for destination, value in enumerate(row)
               if source != destination)


def test_cohort_route_projects_each_destination_bank_and_conserves_work():
    counts = (6, 10, 8, 8)
    raw = _case(route=(counts, True))
    source_rows = (8, 8)
    ops = _ops(raw, tokens=counts[0])
    pack = ops["s0.l0.pack"]
    gate = ops["s0.l0.expert_up"]
    down = ops["s0.l0.expert_down"]
    ledger = dict(pack.metadata)["routing_realization"]

    # The selected graph owns source rank zero; its bank is the destination
    # projection of the complete cohort, not the global ledger and not a
    # fabricated equal split.
    assert pack.local_shape == (counts[0], TOP_K)
    assert ledger["tokens"] == TOKENS
    assert ledger["valid_local_tokens"] == counts[0]
    assert tuple(ledger["local_expert_rows"]) == (8, 8)
    assert tuple(ledger["rank_assignment_rows"]) == (16, 18, 16, 14)
    assert gate.local_shape[0] == sum(source_rows) == 16
    assert down.local_shape[0] == 16
    assert gate.flops == 2 * 16 * gate.local_shape[1] * gate.local_shape[2]
    assert down.flops == 2 * 16 * down.local_shape[1] * down.local_shape[2]
    assert tuple(dict(gate.metadata)["matrix_group_rows"]) == (8, 8)
    assert dict(gate.metadata)["matrix_groups"] == 2

    total_rows = 0
    for rank, expected in enumerate(counts):
        realization = _realization(raw, tokens=expected, rank=rank)
        assert realization.representative_rank == rank
        assert realization.valid_local_tokens == expected
        assert realization.rows == sum(realization.local_expert_rows)
        total_rows += realization.rows
        # Every rank's bank is a real destination partition of the same
        # forward ledger; the sums below are checked once after the loop.
        assert tuple(realization.rank_assignment_rows) == (16, 18, 16, 14)
    assert total_rows == TOKENS * TOP_K
    assert sum(ledger["rank_assignment_rows"]) == TOKENS * TOP_K

    dispatch = ops["s0.l0.dispatch"]
    combine = ops["s0.l0.combine"]
    forward = _matrix(dispatch)
    backward = _matrix(combine)
    assert dispatch.link_bytes == pytest.approx(_off_diagonal(forward) / EP)
    assert combine.link_bytes == pytest.approx(_off_diagonal(backward) / EP)
    # The combine matrix is the dispatch transpose at the BF16 return width.
    assert len(forward) == len(backward) == EP
    for source in range(EP):
        for destination in range(EP):
            assert backward[destination][source] == pytest.approx(
                forward[source][destination])
    assert all(forward[index][index] == 0 for index in range(EP))
    assert dict(dispatch.metadata)["peer_volume_basis"] == "group_mean"
    assert dict(combine.metadata)["peer_volume_basis"] == "group_mean"
    # The declared route has unequal per-source rows; a uniform peer mean would
    # not reproduce the row totals below.
    assert len({sum(row) for row in forward}) > 1


def test_group_rows_and_dispatch_rebind_to_a_different_route_of_the_same_shape():
    first = _case(route=((6, 10, 8, 8), True))
    second = _case(route=((2, 2, 2, 26), True))
    left = _ops(first, tokens=6)["s0.l0.expert_up"]
    right = _ops(second, tokens=6)["s0.l0.expert_up"]
    left_bank = tuple(dict(left.metadata)["matrix_group_rows"])
    right_bank = tuple(dict(right.metadata)["matrix_group_rows"])
    assert left_bank != right_bank
    # Same declared shape, different integer route: the group-list rows come
    # from the current ledger and are not a cached balanced histogram.
    assert sum(left_bank) != sum(right_bank)
    assert left.flops != right.flops
    assert dict(left.metadata)["routing_basis"] == \
        dict(right.metadata)["routing_basis"] == "explicit_token_routing_integer"

    balanced = _ops(_case(balanced=True), tokens=6)["s0.l0.pack"]
    balanced_ledger = dict(balanced.metadata)["routing_realization"]
    assert dict(balanced.metadata)["routing_basis"] == \
        "forced_balanced_integer_counts"
    assert balanced_ledger["token_identities"] == "not_declared"
    assert sum(balanced_ledger["global_expert_rows"]) == 6 * TOP_K
    # The balanced control is explicit, not a silent replacement of the route.
    assert balanced_ledger["basis"] == "forced_balanced_integer_counts"


def test_quantized_producer_rebinds_source_rows_to_the_current_route():
    # Same graph shape and same cohort token total, different current route.
    # Rank 0 owns two tokens in the first case and six in the second: the
    # histogram/quantizer/scale producer must follow the active source ledger,
    # not the graph bucket or a cached shape total.
    first = _case(route=((2, 2, 2, 26), True), tokens=6, quantized=True)
    second = _case(route=((6, 6, 6, 14), True), tokens=6, quantized=True)
    for raw, source_tokens in ((first, 2), (second, 6)):
        ops = _ops(raw, tokens=6)
        histogram = ops["s0.l0.moe_histogram"]
        quantize = ops["s0.l0.moe_transport_quantize"]
        scale = ops["s0.l0.moe_scale_dispatch"]
        payload = ops["s0.l0.moe_payload_dispatch"]
        realization = _realization(raw, tokens=6, rank=0)
        assignments = source_tokens * TOP_K
        assert sum(realization.peer_assignments[0]) == assignments
        assert dict(histogram.metadata)["source_assignments"] == assignments
        assert quantize.local_shape[0] == assignments
        assert dict(quantize.metadata)["source_rows"] == assignments
        assert dict(quantize.metadata)["activation_scale_bytes"] == assignments * 4
        assert quantize.mte2_bytes == assignments * 32 * 2
        assert quantize.mte3_bytes == assignments * 32 + assignments * 4
        assert ops["s0.l0.pack"].mte2_bytes == source_tokens * (32 * 2 + TOP_K * 4)
        assert scale.local_shape[0] == assignments
        assert payload.local_shape[0] == assignments
        # Row scale and INT8 payload rows are the active source rows, never the
        # six-token graph bucket multiplied by top-k.
        assert scale.link_bytes != 12 * 4 * 3 / 4
        assert payload.link_bytes != 12 * 32 * 3 / 4


def test_local_expert_reduce_keeps_a_bound_empty_bank():
    raw = case(tp=2, ep=2, prompt=4, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=1,
        hidden_size=32, moe_intermediate_size=32,
    )
    raw["runtime"].update(moe_recipe=LOCAL_EXPERT_REDUCE,
                          max_num_batched_tokens=16)
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
        "topk_ids": [[0]] * 4, "source_owners": [0] * 4,
        "expert_placement": [[0, 0], [0, 1], [1, 0], [1, 1]],
    }]}
    parsed = parse_case(raw)
    plan, parallel = parsed.model.layer_plans[0], parsed.runtime.parallel
    selected = {}
    for rank in (0, 1):
        realization = _realization(raw, tokens=4, rank=rank)
        assert realization.local_expert_rows == ((4, 0) if rank == 0 else (0, 0))
        ops, last = local_expert_reduce_recipe(
            prefix=f"rank{rank}", plan=plan, deps="input", stage=0,
            requests=("r",), tokens=4, model=parsed.model, parallel=parallel,
            mode="prefill", runtime=parsed.runtime, global_tokens=None,
            routing_realization=realization, chunks=4,
        )
        graph = {op.op_id: op for op in ops}
        selected[rank] = graph
        pack_metadata = dict(graph[f"rank{rank}.pack"].metadata)
        assert pack_metadata["rank_timing_assumption"] == "explicit_physical_rank"
        assert tuple(pack_metadata["represented_ep_ranks"]) == (rank,)
        assert pack_metadata["routing_realization"]["matrix_groups"] == (1 if rank == 0 else 0)
        assert last == f"rank{rank}.moe_output_reduce"
        # pack/unpack/reduce remain real control and state boundaries.
        assert f"rank{rank}.unpack" in graph
        assert f"rank{rank}.moe_output_reduce" in graph
    assert "rank0.expert_up" in selected[0]
    assert "rank1.expert_up" not in selected[1]
    assert "rank1.expert_act" not in selected[1]
    assert selected[0]["rank0.expert_up"].local_shape[0] == 4
    assert selected[0]["rank0.expert_up"].flops > 0
    assert not any(op.recipe in {"grouped_gate_up", "grouped_down", "grouped_swiglu"}
                   for op in selected[1].values())
    assert not any(op.family == "grouped_gemm" for op in selected[1].values())
    # The empty-bank rank still pays its real router/pack/unpack/reduce control
    # work, but no expert grouped FLOP.
    assert "rank1.router" in selected[1]
    assert "rank1.pack" in selected[1]
    # The two physical banks partition the same four assignments.
    assert sum(
        dict(op.metadata)["local_expert_rows"][index]
        for op in (selected[0]["rank0.pack"], selected[1]["rank1.pack"])
        for index in range(2)) == 4


def test_placement_shorthand_binds_to_one_concrete_rank_bank_table():
    forms = (
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [0, [0, 1], 1, [1, 1]],
        [0, 0, 1, 1],
    )
    expected = ((0, 0), (0, 1), (1, 0), (1, 1))
    for placement in forms:
        raw = case(tp=2, ep=2, prompt=4, output=1)
        raw["model"].update(
            layers=1, ffn="moe", num_experts=4, top_k=1,
            hidden_size=32, moe_intermediate_size=32,
        )
        raw["runtime"].update(moe_recipe=LOCAL_EXPERT_REDUCE,
                              max_num_batched_tokens=16)
        raw["runtime"]["routing"] = {"realizations": [{
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
            "topk_ids": [[0]] * 4, "source_owners": [0] * 4,
            "expert_placement": placement,
        }]}
        realization = _realization(raw, tokens=4, rank=0)
        assert realization.expert_placement == expected
        assert realization.to_dict()["expert_placement"] == expected
        assert realization.local_expert_rows == (4, 0)
        # The same concrete table also sorts cleanly for the empty rank.
        empty = _realization(raw, tokens=4, rank=1)
        assert empty.expert_placement == expected
        assert empty.to_dict()["expert_placement"] == expected
        parsed = parse_case(raw)
        ops, _last = local_expert_reduce_recipe(
            prefix="r", plan=parsed.model.layer_plans[0], deps="input",
            stage=0, requests=("r",), tokens=4, model=parsed.model,
            parallel=parsed.runtime.parallel, mode="prefill",
            runtime=parsed.runtime, global_tokens=None,
            routing_realization=realization, chunks=4,
        )
        graph = {op.op_id: op for op in ops}
        assert "r.expert_up" in graph
        assert graph["r.expert_up"].local_shape[0] == 4

    # A genuinely non-contiguous map still fails the local-bank caller.
    raw = case(tp=2, ep=2, prompt=4, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=1,
        hidden_size=32, moe_intermediate_size=32,
    )
    raw["runtime"].update(moe_recipe=LOCAL_EXPERT_REDUCE,
                          max_num_batched_tokens=16)
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 4,
        "topk_ids": [[0]] * 4, "source_owners": [0] * 4,
        "expert_placement": [1, [0, 1], 0, [1, 1]],
    }]}
    parsed = parse_case(raw)
    with pytest.raises(ModelingError, match="contiguous_physical_bank"):
        local_expert_reduce_recipe(
            prefix="r", plan=parsed.model.layer_plans[0], deps="input",
            stage=0, requests=("r",), tokens=4, model=parsed.model,
            parallel=parsed.runtime.parallel, mode="prefill",
            runtime=parsed.runtime, global_tokens=None,
            routing_realization=_realization(raw, tokens=4, rank=0),
            chunks=4,
        )


def test_balanced_full_capture_padding_keeps_logical_and_issued_rows_separate():
    from test_moe_quant_transport import transport_case

    def build(capture):
        raw = transport_case(dp=1, tp=2, ep=2, tokens=1)
        raw["runtime"]["routing"] = {"policy": "balanced"}
        if capture:
            raw["runtime"].update(
                graph_mode="auto", graph_capture_keys=[8],
                graph_capture_mode="full")
        graphs = target_graphs(
            raw, phase="decode", tokens=1, sequences=1, scope="body")
        return {op.op_id: op for op in graphs[0]}

    # Full capture executes the physical eight-row plan even though only one
    # token is logically valid.  Logical validity and issued rows stay distinct.
    padded = build(True)
    pack = padded["s0.l0.pack"]
    histogram = padded["s0.l0.moe_histogram"]
    quantize = padded["s0.l0.moe_transport_quantize"]
    scale = padded["s0.l0.moe_scale_dispatch"]
    payload = padded["s0.l0.moe_payload_dispatch"]
    combine = padded["s0.l0.expert_combine"]
    gate = padded["s0.l0.expert_up"]
    assert pack.local_shape == (8, TOP_K)
    assert pack.mte2_bytes == 8 * (32 * 2 + TOP_K * 4)
    assert pack.mte3_bytes == 8 * TOP_K * (32 * 2 + 4)
    assert dict(histogram.metadata)["source_assignments"] == 8
    assert histogram.mte2_bytes == 8 * 4
    assert histogram.mte3_bytes == 4 * 8  # E4 * int64
    assert quantize.local_shape == (8, 32)
    assert dict(quantize.metadata)["source_rows"] == 8
    assert dict(quantize.metadata)["activation_scale_bytes"] == 8 * 4
    assert quantize.mte2_bytes == 8 * 32 * 2
    assert quantize.mte3_bytes == 8 * 32 + 8 * 4
    assert scale.local_shape == (8,)
    assert payload.local_shape == (8, 32)
    assert scale.link_bytes == 8 * 4 * (2 - 1) / 2
    assert payload.link_bytes == 8 * 32 * (2 - 1) / 2
    assert combine.link_bytes == 8 * 32 * 2 * (2 - 1) / 2
    ledger = dict(gate.metadata)["routing_realization"]
    assert (ledger["valid_local_tokens"], ledger["padded_local_tokens"]) == (1, 8)
    assert sum(ledger["rank_assignment_rows"]) == 8 * TOP_K
    assert dict(histogram.metadata)["source_assignments"] * 2 ==         sum(ledger["rank_assignment_rows"])
    assert dict(gate.metadata)["activation_scale_shape"] == (8,)
    assert dict(gate.metadata)["activation_scale_bytes"] == 8 * 4
    assert gate.local_shape[0] == 8

    # The same plan without capture padding executes one physical row; the
    # unified producer must not keep the padded eight-row total.
    eager = build(False)
    eager_histogram = eager["s0.l0.moe_histogram"]
    eager_quantize = eager["s0.l0.moe_transport_quantize"]
    eager_scale = eager["s0.l0.moe_scale_dispatch"]
    eager_payload = eager["s0.l0.moe_payload_dispatch"]
    eager_gate = eager["s0.l0.expert_up"]
    eager_ledger = dict(eager_gate.metadata)["routing_realization"]
    assert eager_quantize.local_shape == (1, 32)
    assert dict(eager_histogram.metadata)["source_assignments"] == 1
    assert dict(eager_quantize.metadata)["activation_scale_bytes"] == 4
    assert eager_scale.local_shape == (1,)
    assert eager_payload.local_shape == (1, 32)
    assert eager_gate.local_shape[0] == 1
    assert sum(eager_ledger["rank_assignment_rows"]) == 2
    assert dict(eager_histogram.metadata)["source_assignments"] * 2 == 2


def _cross_rank_case():
    raw = case(dp=4, tp=1, ep=4, prompt=1, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=8, top_k=2, hidden_size=32,
        moe_intermediate_size=32, quant=asdict(quant_preset("w8a8_int8")),
    )
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e9
    raw["machine"]["device"].update(
        rates={"memory": 1e9, "vector": 1e9, "mte2_aic": 1e9,
               "mte2_aiv": 1e9, "mte3": 1e9},
        resource_units={"vector": 4, "mte2_aiv": 4, "mte3": 4},
    )
    raw["machine"]["links"][0]["semantics"] = "shared"
    raw["runtime"]["kernels"] = {"moe_transport": {
        "kind": "separate_quantized_payload", "payload_dtype": "int8",
        "scale_dtype": "fp32", "scale_granularity": "row",
        "histogram_dtype": "int64",
    }}
    ids, owners = _ids((6, 10, 8, 8))
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 32,
        "topk_ids": ids, "expert_placement": _placement(),
        "source_owners": owners, "cohort": True,
    }]}
    return raw


def _peer_book(raw, *, capacity=1024, chunk=512):
    book = RecipeBook.from_mapping(raw["runtime"]["kernels"])
    for name in ("expert_quantized_dispatch", "expert_combine",
                 "expert_scale_dispatch"):
        book.recipes[name] = DirectPeerCollective(
            max_inflight_per_rank=1, chunk_bytes=chunk,
            receive_capacity_bytes_per_peer=capacity,
            receive_buffer_id="rx")
    return book


def test_cross_rank_delayed_source_shared_link_and_receive_buffer():
    raw = _cross_rank_case()
    graphs = target_graphs(
        raw, phase="prefill", tokens=1, sequences=(6, 10, 8, 8), scope="body")
    book = _peer_book(raw)

    # Exact issued traffic from the current route, one replica at a time.
    parsed = parse_case(raw)
    for recipe in ("expert_scale_dispatch", "expert_quantized_dispatch",
                   "expert_combine"):
        selected = {r: next(op for op in graphs[r] if op.recipe == recipe)
                    for r in graphs}
        report = cohort_operator_costs(parsed, selected, recipes=book)
        for row in report["operators"]:
            replica = row["replica"]
            matrix = dict(selected[replica].metadata)["peer_traffic"]["payload_bytes"]
            expected = sum(matrix[replica])
            assert row["communication"]["useful_bytes"] == pytest.approx(expected)
            assert row["communication"]["wire_bytes"] == pytest.approx(expected)
            assert row["communication"]["messages"] == 3

    baseline = graph_costs(
        raw, graphs, recipes=book, external_inputs={"analysis_input": 0},
        compact=True)
    baseline_link = baseline["execution"]["serviced"]["link:fabric"]
    assert baseline["execution"]["live_tasks"] == 0
    assert not any(baseline["execution"]["live_allocated"].values())
    assert baseline["execution"]["peak_allocated"]["transport:rx:rx"] == 1024

    # One source is late; route bytes and receiver reservation stay unchanged.
    delayed = TargetGraphSet(
        {r: (tuple(replace(op, deps=(*op.deps, "late_source"))
                   if r == 0 and op.op_id == "s0.l0.pack" else op
                   for op in graphs[r])) for r in graphs},
        kernels=graphs.effective_kernels, selection=graphs.default_selection)
    late = graph_costs(
        raw, delayed, recipes=book,
        external_inputs={"analysis_input": 0, "late_source": 50},
        compact=True)
    assert late["duration_ms"] > baseline["duration_ms"]
    assert late["execution"]["serviced"]["link:fabric"] == pytest.approx(baseline_link)
    assert late["execution"]["peak_allocated"]["transport:rx:rx"] == 1024
    pack = late["profile"]["by_operator"]["s0:s0.l0.pack"]["replicas"]
    assert pack[0]["ready_ms"] >= 50
    assert pack[1]["ready_ms"] < 50
    assert not any(late["execution"]["live_allocated"].values())

    # A second collective shares the same fabric. Its exact payload is added;
    # the delayed route's own issued bytes and receive window are unchanged.
    competing = TargetGraphSet(
        {r: tuple(delayed[r]) + (comm_op(
            op_id="compete_alltoall", deps=("s0.l0.pack",), stage=0, layer=0,
            requests=("r",), family="alltoall", phase="prefill",
            recipe="competing_alltoall", link_bytes=256, shape=(8, 32),
            metadata=(("group_axis", "ep"), ("steps", 3),
                      ("collective_id", "compete"))),) for r in delayed},
        kernels=graphs.effective_kernels, selection=graphs.default_selection)
    book.recipes["competing_alltoall"] = DirectPeerCollective(
        max_inflight_per_rank=1, chunk_bytes=128,
        receive_capacity_bytes_per_peer=512, receive_buffer_id="compete_rx")
    contested = graph_costs(
        raw, competing, recipes=book,
        external_inputs={"analysis_input": 0, "late_source": 50},
        compact=True)
    assert contested["duration_ms"] > late["duration_ms"]
    assert contested["execution"]["serviced"]["link:fabric"] - baseline_link == pytest.approx(4 * 256)
    assert contested["execution"]["peak_allocated"]["transport:rx:rx"] == 1024
    assert contested["execution"]["peak_allocated"]["transport:rx:compete_rx"] > 0
    assert all(value == pytest.approx(0.0, abs=1e-9)
               for value in contested["execution"]["live_allocated"].values())


def test_native_mc2_consumes_the_explicit_route_matrix():
    raw = _case(route=((6, 10, 8, 8), True))
    raw["runtime"]["kernels"] = {
        "native_moe_protocol": {
            "kind": "arch35_fullmesh", "max_inflight_per_rank": 4}}
    ops = _ops(raw, tokens=6)
    dispatch = dict(ops["s0.l0.dispatch"].metadata)["mc2_phase"]
    combine = dict(ops["s0.l0.unpack"].metadata)["mc2_phase"]
    peer = _realization(raw, tokens=6, rank=0).peer_assignments
    assert dispatch["route_matrix"] == peer
    assert combine["route_matrix"] == peer
    assert dispatch["route_reverse"] is False
    assert combine["route_reverse"] is True
    assert dispatch["route_rank"] == combine["route_rank"] == 0
    assert dispatch["payload_basis"] == \
        combine["payload_basis"] == "explicit_token_routing_integer"
    # The source view is six local tokens; the received view is the complete
    # destination bank partition. They are intentionally different ledgers.
    assert dispatch["source_assignments"] == 12
    assert combine["source_assignments"] == 16


def test_cohort_subgraph_executes_once_and_conserves_total_work():
    from mindie_sim import graph_costs, target_graphs

    counts = (6, 10, 8, 8)
    raw = _case(route=(counts, True))
    graphs = target_graphs(
        raw, phase="prefill", tokens=counts[0], sequences=1, scope="body")
    report = graph_costs(
        raw, graphs, external_inputs={"analysis_input": 0})
    ops = graphs[0]
    work = report["execution"]["work"]
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    # Every active operator is submitted once; the subgraph total is the sum
    # of its own declared work, not a second model-level fit or a per-rank
    # multiplication of the representative graph.
    assert work["flops"] == pytest.approx(sum(op.flops for op in ops))
    assert work["hbm_bytes"] == pytest.approx(sum(op.hbm_bytes for op in ops))
    assert work["link_bytes"] == pytest.approx(sum(op.link_bytes for op in ops))
    # The actual directed peer matrix is larger than the scalar group mean on
    # purpose: the engine keeps per-source cells while the aggregate work
    # ledger remains the mean boundary. Both are visible.
    dispatch = next(op for op in ops if op.recipe == "expert_dispatch")
    combine = next(op for op in ops if op.recipe == "expert_combine")
    link = report["execution"]["serviced"].get("link:fabric", 0.0)
    assert link > dispatch.link_bytes + combine.link_bytes
    assert sum(map(sum, _matrix(dispatch))) == 2944
    assert sum(map(sum, _matrix(combine))) == 2944


def test_quantized_transport_keeps_scale_rows_and_the_actual_rank_matrix():
    counts = (4, 12, 8, 8)
    raw = _case(route=(counts, True), quantized=True)
    ops = _ops(raw, tokens=counts[0])
    quantize = ops["s0.l0.moe_transport_quantize"]
    scale = ops["s0.l0.moe_scale_dispatch"]
    payload = ops["s0.l0.moe_payload_dispatch"]
    combine = ops["s0.l0.expert_combine"]

    # Source rank zero owns four tokens * top-k two = eight assignment rows.
    assert dict(quantize.metadata)["source_rows"] == 8
    assert dict(quantize.metadata)["activation_scale_shape"] == (8,)
    assert dict(quantize.metadata)["activation_scale_bytes"] == 8 * 4
    assert dict(quantize.metadata)["output_format"] == "int8"
    assert dict(quantize.metadata)["moe_peer_rank"] == 0
    assert dict(quantize.metadata)["moe_peer_basis"] == \
        "explicit_token_routing_integer"
    # The permutation/inverse edges and the quantized scale producer stay
    # attached to the same pack/combine pair.
    pack = ops["s0.l0.pack"]
    gate = ops["s0.l0.expert_up"]
    unpack = ops["s0.l0.unpack"]
    assert quantize.deps == (pack.op_id,)
    assert set(gate.deps) == {quantize.op_id, scale.op_id, payload.op_id}
    assert combine.op_id in unpack.deps
    assert pack.op_id in unpack.deps

    scale_matrix = _matrix(scale)
    payload_matrix = _matrix(payload)
    combine_matrix = _matrix(combine)
    # Row scales are FP32; the int8 payload and BF16 return have distinct
    # widths.  The assignment counts are identical across all three ledgers.
    assert all(value % 4 == 0 for row in scale_matrix for value in row)
    for source in range(EP):
        for destination in range(EP):
            assert payload_matrix[source][destination] == \
                scale_matrix[source][destination] * 8
            assert combine_matrix[destination][source] == \
                scale_matrix[source][destination] * 16
    assert _off_diagonal(scale_matrix) / 4 == pytest.approx(scale.link_bytes)
    assert _off_diagonal(payload_matrix) / 4 == pytest.approx(payload.link_bytes)
    assert _off_diagonal(combine_matrix) / 4 == pytest.approx(combine.link_bytes)
    assert dict(scale.metadata)["peer_volume_basis"] == "group_mean"
    assert dict(payload.metadata)["peer_volume_basis"] == "group_mean"
    assert dict(combine.metadata)["peer_volume_basis"] == "group_mean"
    # The return is the transpose of the forward assignment matrix; exact
    # integer cells make the unequal source totals visible.
    for source in range(EP):
        for destination in range(EP):
            assert combine_matrix[destination][source] == pytest.approx(
                payload_matrix[source][destination] * 2.0)
    assert len({sum(row) for row in payload_matrix}) > 1
    # The direct inverse keeps its matrix and does not need a second binding
    # pass that only averages the same cells back to a uniform scalar.
    assert "reverse_of_collective" not in dict(combine.metadata)

    # The same explicit route is a runnable subgraph: one submission of each
    # active operator, no live task and no allocation left behind.
    graphs = target_graphs(
        raw, phase="prefill", tokens=counts[0], sequences=1, scope="body")
    report = graph_costs(raw, graphs, external_inputs={"analysis_input": 0})
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    assert report["execution"]["work"]["link_bytes"] == pytest.approx(
        sum(op.link_bytes for op in graphs[0]))
    assert not any(report["execution"]["live_allocated"].values())
    assert report["execution"]["serviced"].get("link:fabric", 0.0) > 0


def test_local_expert_reduce_uses_the_bound_rank_bank_and_rebinds_rows():
    counts = (6, 10, 8, 8)
    raw = _case(route=(counts, False), tokens=TOKENS)
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    parsed = parse_case(raw)
    plan = parsed.model.layer_plans[0]
    parallel = parsed.runtime.parallel
    rows = []
    for rank, expected_tokens in enumerate(counts):
        realization = _realization(raw, tokens=TOKENS, rank=rank)
        assert realization.representative_rank == rank
        options = dict(
            prefix=f"rank{rank}", plan=plan, deps="input", stage=0,
            requests=("r",), tokens=TOKENS, model=parsed.model,
            parallel=parallel, mode="prefill", runtime=parsed.runtime,
            global_tokens=None, routing_realization=realization, chunks=4,
        )
        ops, last = local_expert_reduce_recipe(**options)
        graph = {op.op_id: op for op in ops}
        gate = graph[f"rank{rank}.expert_up"]
        metadata = dict(gate.metadata)
        assert metadata["routing_basis"] == "explicit_token_routing_integer"
        assert tuple(metadata["local_expert_rows"]) == \
            tuple(realization.local_expert_rows)
        assert gate.local_shape[0] == realization.rows
        assert metadata["matrix_groups"] == realization.active
        assert metadata["histogram_counts"] == \
            tuple(realization.local_expert_rows)
        # Router allocation stays the replicated source width; only the local
        # expert bank is filtered.
        assert graph[f"rank{rank}.pack"].local_shape == (TOKENS, TOP_K)
        assert last == f"rank{rank}.moe_output_reduce"
        rows.append(realization.rows)
    assert sum(rows) == TOKENS * TOP_K
    assert rows == [16, 18, 16, 14]

    # A different route of the same shape rebinds the local group list.
    other = _case(route=((2, 2, 2, 26), False), tokens=TOKENS)
    changed = _realization(other, tokens=TOKENS, rank=0)
    assert changed.rows != rows[0]


def test_non_contiguous_physical_placement_is_rejected_by_the_local_bank_caller():
    raw = _case(route=((6, 10, 8, 8), False), tokens=TOKENS)
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    # Swap two banks while keeping the same physical rank set: exact routing
    # still describes a legal exchange, but it is not one contiguous weight
    # bank, so the local-reduce caller must fail rather than relabel it.
    swapped = [list(pair) for pair in _placement()]
    swapped[0], swapped[2] = swapped[2], swapped[0]
    raw["runtime"]["routing"]["realizations"][0]["expert_placement"] = swapped
    parsed = parse_case(raw)
    with pytest.raises(ModelingError, match="contiguous_physical_bank"):
        local_expert_reduce_recipe(
            prefix="rank0", plan=parsed.model.layer_plans[0], deps="input",
            stage=0, requests=("r",), tokens=TOKENS, model=parsed.model,
            parallel=parsed.runtime.parallel, mode="prefill",
            runtime=parsed.runtime, global_tokens=None,
            routing_realization=_realization(raw, tokens=TOKENS, rank=0),
            chunks=4,
        )


def _service_route_case(ids):
    raw = case(tp=4, ep=4, prompt=32, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=8, top_k=2,
        hidden_size=32, moe_intermediate_size=32,
    )
    owners = []
    for source, count in enumerate((6, 10, 8, 8)):
        owners.extend([source] * count)
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 32,
        "topk_ids": ids, "expert_placement": _placement(),
        "source_owners": owners, "cohort": True,
    }]}
    return raw


def test_same_shape_two_routes_rebind_in_model_and_service():
    source_ids, _owners = _ids((6, 10, 8, 8))
    # Same 32-token shape and same source ownership; only the live route changes.
    local_ids = [[0, 1]] * 32
    first_raw = _service_route_case(source_ids)
    second_raw = _service_route_case(local_ids)
    first = predict(first_raw)
    second = predict(second_raw)
    assert first["status"] == second["status"] == "modeled"
    assert first["requests"][0]["ttft_ms"] != second["requests"][0]["ttft_ms"]
    assert first["execution"]["live_tasks"] == second["execution"]["live_tasks"] == 0
    assert not any(first["execution"]["live_allocated"].values())
    assert not any(second["execution"]["live_allocated"].values())
    # The two services bind the current route, not the previous shape total.
    assert first["execution"]["work"] != second["execution"]["work"]
    assert first["traces"][0]["operators"] == second["traces"][0]["operators"]

    view = RequestView("r", 32, 0, True, "prefill")
    parsed = parse_case(first_raw)
    realization = select_realization(
        parsed.runtime, parsed.model.layer_plans[0], phase="prefill",
        tokens=32, valid_tokens=32, global_tokens=None, views=(view,),
        physical_rank=0, query_position=0)
    assert realization.topk_ids == tuple(tuple(row) for row in source_ids)
    assert realization.valid_local_tokens == 6
    # T14 handoff: exact current RequestView plus the selected realization.
    handoff = {
        "request_view": {
            "request_id": view.request_id, "q_len": view.q_len,
            "context": view.context, "phase": view.phase,
            "samples": view.samples, "query_ranges": view.query_ranges,
        },
        "realization": realization.to_dict(),
        "service": {
            "ttft_ms": first["requests"][0]["ttft_ms"],
            "work": first["execution"]["work"],
            "live_tasks": first["execution"]["live_tasks"],
        },
    }
    assert handoff["realization"]["topk_ids"] == tuple(tuple(row) for row in source_ids)
    assert handoff["service"]["live_tasks"] == 0


def test_real_a5_source_force_route_identity_binds_to_a_runnable_slice():
    from mindie_sim.prediction.source_force_eplb import SOURCE, SourceForceEplb
    from test_source_force_eplb import case as source_case

    # This is the existing A5 Kimi source-cursor scenario, not a synthetic id
    # table.  Its source revision/cursor/mask identity travels with the ledger.
    raw = source_case()
    ops = {op.op_id: op for op in target_operators(
        raw, phase="decode", tokens=1, context=4104, sequences=4, scope="body")}
    gate = ops["s0.l1.expert_up"]
    routing = dict(gate.metadata)["routing_realization"]
    assert routing["basis"] == "source_force_eplb_masked_integer_counts"
    assert routing["source"] == SOURCE
    assert routing["source"]["repository"] == "vllm-project/vllm-ascend"
    assert routing["source"]["revision"] == "8c30e4f491f73cba737a1dacc522894e437cdf3d"
    assert routing["source"]["cursor"] == "vllm_ascend/ops/fused_moe/force_eplb.py:130-142"
    peer = routing["peer_assignment_rows"]
    assert len(peer) == 8
    assert sum(map(sum, peer)) == routing["valid_global_tokens"] * 8
    assert sum(routing["rank_assignment_rows"]) == sum(map(sum, peer))
    assert sum(routing["local_expert_rows"]) == gate.local_shape[0]
    dispatch = dict(ops["s0.l1.dispatch"].metadata)["mc2_phase"]
    combine = dict(ops["s0.l1.unpack"].metadata)["mc2_phase"]
    assert dispatch["payload_basis"] == "source_force_eplb_masked_integer_counts"
    assert dispatch["route_matrix"] == peer
    assert combine["route_matrix"] == peer
    assert dispatch["source_assignments"] == 8
    assert combine["source_assignments"] == 4
    # Both directions keep the source/destination assignment total.
    assert sum(sum(row) for row in peer) == 32

    # Concrete T12 handoff: the source-cursor expansion names every valid
    # token, its source owner and physical bank, while keeping the scenario
    # label separate from a captured binary identity.
    route = SourceForceEplb(
        expert_rows=tuple(routing["global_expert_rows"]),
        rank_expert_rows=tuple(tuple(row) for row in routing["rank_expert_rows"]),
        peer_assignments=tuple(tuple(row) for row in peer),
        valid_tokens_per_rank=tuple(routing["valid_tokens_per_rank"]),
        allocated_tokens_per_rank=int(routing["allocated_tokens_per_rank"]),
        top_k=int(routing["top_k"]),
        ep=int(routing["ep"]),
        valid_global_tokens=int(routing["valid_global_tokens"]),
        representative_rank=int(routing["representative_rank"]),
    )
    occurrence = route.typed_occurrence(payload_scale_producer={
        "dispatch_op": ops["s0.l1.dispatch"].op_id,
        "payload_dtype": dispatch["dtype"],
        "scale_dtype": None,
        "width": dispatch["width"],
        "source_tokens": dispatch["source_tokens"],
        "received_assignments": dispatch["received_assignments"],
        "operator": "mc2_protocol",
    })
    assert occurrence["source"] == SOURCE
    assert occurrence["capture_identity"].startswith("not_established")
    assert occurrence["token_order_basis"] == "derived_from_source_cursor"
    assert len(occurrence["token_order"]) == 4
    assert occurrence["source_owners"] == (0, 1, 2, 3)
    assert occurrence["expert_placement"] == tuple(
        (expert // 2, expert % 2) for expert in range(16))
    assert occurrence["source_destination_rows"] == peer
    assert occurrence["destination_bank_rows"] == routing["rank_expert_rows"]
    assert occurrence["model_top_k"] == 8
    assert occurrence["native_top_k"] == 1
    assert occurrence["payload_scale_producer"]["dispatch_op"] == "s0.l1.dispatch"


def test_family_align_and_combine_keep_explicit_ids_and_local_rows():
    raw = case(tp=1, ep=1, prompt=8, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=2,
        hidden_size=32, moe_intermediate_size=32,
    )
    ids = [[0, 1], [0, 2], [1, 2], [0, 1],
           [2, 3], [1, 3], [0, 3], [2, 3]]
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "tokens": 8,
        "topk_ids": ids, "source_owners": [0] * 8,
        "expert_placement": [[0, 0], [0, 1], [0, 2], [0, 3]],
    }]}
    raw["runtime"]["kernels"] = {"lowerings": {
        "route_pack": {
            "implementation": "cuda_moe_align_block_size",
            "options": {"block_size": 16, "pad_sorted_ids": True},
        },
        "route_unpack": {
            "implementation": "triton_moe_fused_mul_sum",
            "options": {"weight_dtype": "bf16", "compute_capability": 89},
        },
    }}
    ops = _ops(raw, tokens=8)
    align = ops["s0.l0.pack.align"]
    materialize = ops["s0.l0.unpack.materialize"]
    combine = ops["s0.l0.unpack"]
    align_meta = dict(align.metadata)
    assert align_meta["routing_basis"] == "explicit_token_routing_integer"
    assert align_meta["semantic_layout"] == "routing_indices_only"
    expected_counts = [0] * 4
    for row in ids:
        for expert in row:
            expected_counts[expert] += 1
    assert tuple(align_meta["analysis"]["counts"]) == tuple(expected_counts)
    materialize_meta = dict(materialize.metadata)
    realization = materialize_meta["routing_realization"]
    assert realization["basis"] == "explicit_token_routing_integer"
    assert tuple(materialize_meta["expert_rows"]) == \
        tuple(realization["local_expert_rows"])
    assert sum(materialize_meta["expert_rows"]) == \
        materialize_meta["input_rows"] == 16
    assert dict(combine.metadata)["routing_realization"]["basis"] == \
        "explicit_token_routing_integer"
