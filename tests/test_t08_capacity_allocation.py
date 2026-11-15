"""T08 allocation/access counterexamples and A5 capacity slices.

The tests use independent integer geometry for the H2/H3 controls and a real
catalog A5 deployment for the public capacity report.  They do not assert a
kernel duration, a table keyed by shape, or an automatic dtype fallback.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.catalog.cache import cache_row_split, layer_cache_components
from mindie_sim.catalog.cache_pages import CachePageGroup, KVPageAllocation
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.capacity import (
    Allocation, AllocationPlan, Memory, MemoryPolicy,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect
from test_cache_pages import compressed_case
from test_prediction_system import case, linear_work
from test_sparse_kv_offload import _glm_sparse_case

ROOT = Path(__file__).resolve().parents[1]
A5_QWEN = ROOT / "examples/cases/qwen38-a5.json"
A5_GLM = ROOT / "examples/cases/glm53-a5.json"
A5_V41 = ROOT / "examples/cases/v41-a5-dp4.json"
MIB = 1024 * 1024


def test_h2_page_allocation_separates_payload_tail_scale_metadata_and_access():
    bf16 = KVPageAllocation(
        token_spans=(17, 31), page_tokens=16, local_kv_heads=1, head_dim=8,
        kv_count=2, element_bytes=2, metadata_bytes_per_page=32,
    )
    assert bf16.logical_tokens == 48
    assert bf16.pages == 4              # per-owner pages, not ceil(48 / 16)
    assert bf16.physical_token_slots == 64
    assert bf16.valid_payload_bytes == 1536
    assert bf16.allocated_payload_bytes == 2048
    assert bf16.page_tail_payload_bytes == 512
    assert bf16.scale_bytes == 0
    assert bf16.metadata_bytes == 128
    assert bf16.total_allocation_bytes == 2176
    # Allocation is not access: only valid rows plus declared page metadata.
    assert bf16.access_bytes == 1664
    assert bf16.access_payload_bytes + bf16.access_metadata_bytes == 1664

    int8 = KVPageAllocation(
        token_spans=(17, 31), page_tokens=16, local_kv_heads=1, head_dim=8,
        kv_count=2, element_bytes=1, scale_bytes_per_page=8,
        metadata_bytes_per_page=32,
    )
    assert int8.valid_payload_bytes == 768
    assert int8.allocated_payload_bytes == 1024
    assert int8.page_tail_payload_bytes == 256
    assert int8.scale_bytes == 32       # one FP32 K + one FP32 V per page
    assert int8.metadata_bytes == 128
    assert int8.total_allocation_bytes == 1184
    assert int8.access_bytes == 928
    assert int8.access_scale_bytes == 32
    # The padded payload is capacity, never an access.
    assert int8.access_payload_bytes == int8.valid_payload_bytes
    assert int8.access_payload_bytes != int8.allocated_payload_bytes

    with pytest.raises(ModelingError, match="kv_page_allocation_requires_positive_geometry"):
        KVPageAllocation((1,), page_tokens=0)
    with pytest.raises(ModelingError, match="kv_page_allocation_invalid:element_bytes"):
        KVPageAllocation((1,), element_bytes=-1)


def test_cache_row_scale_is_part_of_allocation_not_a_bitwidth_multiplier():
    parsed = parse_case(case())
    model = replace(parsed.model, metadata=(("cache_representations", {
        "mla_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
    }),))
    payload, scale = cache_row_split(model, "mla_kv", 16)
    from mindie_sim.catalog.cache_format import cache_format
    assert payload == 16
    assert scale == 2 * 4       # two eight-value groups, one FP32 each
    assert payload + scale == cache_format(model, "mla_kv").row_bytes(16)
    # The same width with BF16 has no invented scale bytes.
    bf16_payload, bf16_scale = cache_row_split(parsed.model, "mla_kv", 16)
    assert bf16_payload == 32 and bf16_scale == 0


def _joint_plan(kv_bytes: float, workspace_bytes: float, *, temporary: float = 0.0,
                workspace_copies: int = 1, sequential_workspace: bool = False) -> AllocationPlan:
    allocations = [
        Allocation("weights", 536870912, "weights"),
        Allocation("runtime_persistent", 67108864, "runtime"),
        # A pool reservation is charged once; its suballocations are not added.
        Allocation("communication_pool", 67108864, "pool"),
        Allocation("communication_tx", 32 * MIB, "communication", pool="communication_pool"),
        Allocation("communication_rx", 32 * MIB, "communication", pool="communication_pool"),
        # A view of an existing allocation is not a second allocation.
        Allocation("weights_view", 4096, "workspace", alias_of="weights"),
        Allocation("kv", kv_bytes, "cache"),
        Allocation("workspace", workspace_bytes, "workspace"),
        Allocation("temporary", temporary, "temporary"),
    ]
    if workspace_copies == 2:
        allocations.append(Allocation(
            "workspace_reuse", workspace_bytes, "workspace",
            live=not sequential_workspace))
    return AllocationPlan(tuple(allocations), capacity_bytes=1024 ** 3)


def test_h3_joint_capacity_fixed_plan_retains_infeasibility_and_concurrency():
    expected = [
        ("bf16_workspace128", _joint_plan(268566528, 134217728), 1073872896, False),
        ("bf16_workspace96", _joint_plan(268566528, 100663296), 1040318464, True),
        ("int8_workspace128", _joint_plan(134873088, 134217728), 940179456, True),
        ("int8_two_concurrent_workspace128",
         _joint_plan(134873088, 134217728, workspace_copies=2), 1074397184, False),
        ("int8_workspace128_and_bf16_temporary",
         _joint_plan(134873088, 134217728, temporary=268435456), 1208614912, False),
    ]
    for name, plan, peak, feasible in expected:
        report = plan.to_dict()
        assert report["peak_bytes"] == peak, name
        assert report["feasible"] is feasible, name
        # Alias and pool bytes are reported but never added to the peak.
        assert report["alias_bytes_not_added"] == 4096
        assert report["pool_bytes_not_added"] == 64 * MIB
        assert report["categories"]["pool"] == 67108864

    # Two separately live workspace buffers overlap; sequential reuse has one.
    concurrent = _joint_plan(134873088, 134217728, workspace_copies=2,
                             sequential_workspace=False)
    sequential = _joint_plan(134873088, 134217728, workspace_copies=2,
                             sequential_workspace=True)
    assert concurrent.resolve()["peak_bytes"] - sequential.resolve()["peak_bytes"] == 134217728
    assert sequential.resolve()["retired_sequential_bytes"] == 134217728
    with pytest.raises(ValueError, match="allocation_pool_overcommitted"):
        AllocationPlan((
            Allocation("pool", 1024, "pool"),
            Allocation("slice", 2048, "communication", pool="pool"),
        )).resolve()

    # A fixed plan is retained as infeasible; there is no fallback field/switch.
    fixed = _joint_plan(268566528, 134217728).to_dict()
    assert fixed["feasible"] is False
    assert fixed["categories"]["workspace"] == 134217728


def test_page_metadata_is_charged_once_and_enters_real_admission():
    raw = case(prompt=4, output=1)
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    layer = parsed.model.layer_plans[0]
    base_total = base.cache_bytes(3)[0]
    layer_total = sum(__import__(
        "mindie_sim.catalog.cache", fromlist=["layer_cache_components"]
    ).layer_cache_components(
        parsed.model, layer, parsed.runtime.parallel, 3, capacity_padded=True,
        interleave=parsed.runtime.kv_block_size,
    ).values())
    other_total = base_total - layer_total
    group = {"layers": [layer.index], "tokens_per_page": 4,
             "bytes_per_layer_page": 256, "metadata_bytes_per_page": 32}
    declared = CachePageGroup.from_mapping(group)
    assert declared.pages(3, 128) == 1
    assert declared.metadata_bytes(1) == 32
    # A per-layer declaration owns one metadata record per covered layer.
    two_layer = CachePageGroup.from_mapping({
        "layers": [0, 1], "tokens_per_page": 4,
        "bytes_per_layer_page": 64, "metadata_bytes_per_page": 32,
    })
    assert two_layer.metadata_bytes(1) == 64
    group_page = CachePageGroup.from_mapping({
        "layers": [0, 1], "tokens_per_page": 4,
        "bytes_per_group_page": 128, "metadata_bytes_per_page": 32,
    })
    assert group_page.metadata_bytes(1) == 32
    parsed_group = replace(parsed, runtime=replace(
        parsed.runtime, memory={"derive_workspace": False,
                                "cache_page_groups": [group]}))
    with_group = Memory(parsed_group)
    # One 4-token page, 64 payload + one 32-byte metadata record; other layers
    # keep their own allocations.
    expected = other_total + 256 + 32
    assert with_group.cache_bytes(3) == (expected,)
    allocation = with_group.cache_allocation(3)
    assert allocation["allocated"]["metadata_bytes_per_stage_rank"] == [32.0]
    valid_total = with_group.cache_bytes(3, padded=False)[0]
    assert allocation["page_tail"]["total_bytes_per_stage_rank"] == [
        expected - valid_total - allocation["allocated"][
            "metadata_bytes_per_stage_rank"][0]]
    assert allocation["page_tail"]["total_bytes_per_stage_rank"][0] > 0.0

    # Capacity equal to the allocation without page metadata is insufficient
    # once the declared metadata is admitted; no dtype/algorithm fallback.
    tight = replace(with_group.case, machine=replace(
        with_group.case.machine,
        dies=tuple(replace(die, memory_bytes=int(with_group.weights[0] + other_total + 256))
                   for die in with_group.case.machine.dies)))
    rejected = Memory(tight)
    assert rejected.resize(0, "r", 3) is False
    assert rejected.live == [[0.0]]

    exact = replace(with_group.case, machine=replace(
        with_group.case.machine,
        dies=tuple(replace(die, memory_bytes=int(with_group.weights[0] + expected))
                   for die in with_group.case.machine.dies)))
    admitted = Memory(exact)
    assert admitted.resize(0, "r", 3) is True
    assert admitted.live == [[expected]]


def test_concurrent_workspace_copies_and_temporary_dequant_change_admission():
    parsed = parse_case(case())
    base = Memory(parsed, MemoryPolicy(derive_workspace=True,
                                       concurrent_workspace_copies=1))
    doubled = Memory(parsed, MemoryPolicy(derive_workspace=True,
                                         concurrent_workspace_copies=2))
    assert doubled.workspace == [2 * value for value in base.workspace]
    assert doubled.capacity == [[cap - value for cap, value in zip(
        base.capacity[0], base.workspace)]]

    temporary = Memory(parsed, MemoryPolicy(
        derive_workspace=True, concurrent_workspace_copies=1,
        temporary_dequant_bytes=12345))
    assert temporary.temporary_dequant == [12345]
    assert temporary.capacity == [[cap - 12345 for cap in base.capacity[0]]]
    report = temporary.to_dict()
    assert report["temporary_dequant_bytes_per_stage_rank"] == [12345]
    assert report["fixed_plan"] is True
    assert report["fallback"] == "none"
    with pytest.raises(ValueError, match="concurrent_workspace_copies"):
        Memory(parsed, MemoryPolicy(concurrent_workspace_copies=0))
    with pytest.raises(ValueError, match="temporary_dequant_bytes"):
        Memory(parsed, MemoryPolicy(temporary_dequant_bytes=-1))


def test_quantization_and_reserve_declarations_reach_allocation_geometry():
    parsed = parse_case(case())
    bf16 = Memory(parsed, MemoryPolicy(derive_workspace=False))
    quant = replace(bf16.case.model.quant or QuantSpec("bf16"), name="int8-kv",
                    kv_dtype="int8")
    int8 = Memory(replace(bf16.case, model=replace(bf16.case.model, quant=quant)),
                  MemoryPolicy(derive_workspace=False))
    assert int8.cache_bytes(8)[0] < bf16.cache_bytes(8)[0]
    assert int8.cache_allocation(8)["allocated"]["total_bytes_per_stage_rank"][0] < \
        bf16.cache_allocation(8)["allocated"]["total_bytes_per_stage_rank"][0]
    # Persistent capacity is unchanged, but the same span now admits under a
    # device capacity that rejected the BF16 cache representation.
    tight = replace(bf16.case, machine=replace(
        bf16.case.machine,
        dies=tuple(replace(die, memory_bytes=int(
            bf16.weights[0] + bf16.cache_bytes(8)[0] - 1))
            for die in bf16.case.machine.dies)))
    bf16_tight = Memory(tight, MemoryPolicy(derive_workspace=False))
    int8_tight = Memory(replace(tight, model=replace(tight.model, quant=quant)),
                        MemoryPolicy(derive_workspace=False))
    assert bf16_tight.resize(0, "r", 8) is False
    assert int8_tight.resize(0, "r", 8) is True

    graph = Memory(parsed, MemoryPolicy(derive_workspace=False,
                                        graph_pool_bytes=7 * MIB))
    assert graph.capacity[0][0] == bf16.capacity[0][0] - 7 * MIB
    assert graph.to_dict()["reserve_breakdown"]["graph_pool_bytes"] == 7 * MIB


def test_gqa_explicit_representation_drives_total_and_admission():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    plan = replace(parsed.model.layer_plans[0], num_kv_heads=1, head_dim=8)
    runtime = replace(parsed.runtime, pp_layer_partition=(1,))
    default_model = replace(parsed.model, layers=1, layer_plans=(plan,))
    default = Memory(replace(parsed, model=default_model, runtime=runtime))
    explicit_model = replace(default_model, metadata=((
        "cache_representations", {
            "gqa_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
        }),))
    explicit = Memory(replace(parsed, model=explicit_model, runtime=runtime))
    # One 16-element row: BF16 32 B; INT8 payload 16 B + two FP32 scales 8 B.
    assert default.cache_bytes(1) == (32.0,)
    assert explicit.cache_bytes(1) == (24.0,)
    allocation = explicit.cache_allocation(1)
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [16.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [8.0]
    assert allocation["allocated"]["total_bytes_per_stage_rank"] == [24.0]
    assert allocation["page_tail"]["total_bytes_per_stage_rank"] == [0.0]

    # Capacity with 24 B of KV admits the explicit representation and rejects
    # the default 32 B row; the row representation is not a post-hoc ratio.
    tight = replace(explicit.case, machine=replace(
        explicit.case.machine,
        dies=tuple(replace(die, memory_bytes=int(default.weights[0] + 24))
                   for die in default.case.machine.dies)))
    explicit_tight = Memory(tight)
    default_tight = Memory(replace(tight, model=default_model))
    assert default_tight.resize(0, "r", 1) is False
    assert explicit_tight.resize(0, "r", 1) is True
    assert explicit_tight.live == [[24.0]]


def test_group_page_fills_one_physical_page_independent_of_generic_kv_block():
    raw = case()
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    plan = replace(parsed.model.layer_plans[0], num_kv_heads=1, head_dim=8)
    runtime = replace(parsed.runtime, pp_layer_partition=(1,), kv_block_size=8)
    model = replace(parsed.model, layers=1, layer_plans=(plan,), metadata=((
        "cache_representations", {
            "gqa_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
        }),))
    group = {"layers": [0], "components": ["gqa_kv"],
             "tokens_per_page": 3, "bytes_per_group_page": 72}
    memory = Memory(replace(parsed, model=model, runtime=replace(
        runtime, memory={"derive_workspace": False, "cache_page_groups": [group]})))
    allocation = memory.cache_allocation(3)
    # One fully filled 3-token page has 48 B payload + 24 B scale; generic
    # kv_block_size=8 must not create extra rows/scale or a negative tail.
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [48.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [24.0]
    assert allocation["allocated"]["total_bytes_per_stage_rank"] == [72.0]
    assert allocation["valid"]["payload_bytes_per_stage_rank"] == [48.0]
    assert allocation["valid"]["scale_bytes_per_stage_rank"] == [24.0]
    assert allocation["page_tail"]["total_bytes_per_stage_rank"] == [0.0]
    assert memory.cache_bytes(3) == (72.0,)
    # One valid row still occupies the same three-row physical page.
    one_row = memory.cache_allocation(1)
    assert one_row["allocated"]["payload_bytes_per_stage_rank"] == [48.0]
    assert one_row["allocated"]["scale_bytes_per_stage_rank"] == [24.0]
    assert one_row["allocated"]["total_bytes_per_stage_rank"] == [72.0]
    assert one_row["valid"]["payload_bytes_per_stage_rank"] == [16.0]
    assert one_row["valid"]["scale_bytes_per_stage_rank"] == [8.0]
    assert one_row["page_tail"]["payload_bytes_per_stage_rank"] == [32.0]
    assert one_row["page_tail"]["scale_bytes_per_stage_rank"] == [16.0]
    assert one_row["page_tail"]["total_bytes_per_stage_rank"] == [48.0]
    assert memory.cache_bytes(1) == (72.0,)
    # Same physical page with a different generic block still reports 72 B.
    other_block = replace(runtime, kv_block_size=1, memory={
        "derive_workspace": False, "cache_page_groups": [group]})
    other = Memory(replace(parsed, model=model, runtime=other_block))
    assert other.cache_bytes(3) == (72.0,)
    assert other.cache_allocation(3)["allocated"]["total_bytes_per_stage_rank"] == [72.0]


def test_window_group_metadata_uses_resident_window_pages_not_full_sequence():
    parsed = compressed_case()
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "layers": [0, 1], "components": ["window"],
            "tokens_per_page": 3, "bytes_per_group_page": 96,
            "resident_tokens": 5, "metadata_bytes_per_page": 32,
        }],
    })
    memory = Memory(replace(parsed, runtime=runtime))
    group = memory.pages.groups[0]
    assert group.pages(9, parsed.runtime.max_model_len) == 2
    assert group.touched_pages(9, parsed.runtime.max_model_len) == 2
    allocation = memory.cache_allocation(9)
    # Six physical rows per layer, five valid rows; window truncation applies
    # only to valid content, not to allocated payload capacity.
    assert allocation["page_groups"]["component_payload_bytes_per_stage_rank"] == [192.0]
    assert allocation["page_groups"]["valid_component_payload_bytes_per_stage_rank"] == [160.0]
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [352.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [0.0]
    assert allocation["allocated"]["reserve_bytes_per_stage_rank"] == [0.0]
    assert allocation["valid"]["payload_bytes_per_stage_rank"] == [320.0]
    assert allocation["valid"]["scale_bytes_per_stage_rank"] == [0.0]
    assert allocation["page_tail"]["payload_bytes_per_stage_rank"] == [32.0]
    assert allocation["allocated"]["metadata_bytes_per_stage_rank"] == [64.0]
    assert allocation["access"]["metadata_bytes_per_stage_rank"] == [64.0]
    assert allocation["page_tail"]["metadata_bytes_per_stage_rank"] == [0.0]


def test_window_page_allocates_full_slots_while_valid_is_window_limited():
    raw = case()
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    plan = replace(parsed.model.layer_plans[0], num_kv_heads=1, head_dim=8,
                   window=5)
    runtime = replace(parsed.runtime, pp_layer_partition=(1,), kv_block_size=8)
    model = replace(parsed.model, layers=1, layer_plans=(plan,), metadata=((
        "cache_representations", {
            "gqa_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
        }),))
    group = {"layers": [0], "components": ["gqa_kv"],
             "tokens_per_page": 3, "bytes_per_group_page": 72,
             "resident_tokens": 5, "metadata_bytes_per_page": 32}
    memory = Memory(replace(parsed, model=model, runtime=replace(
        runtime, memory={"derive_workspace": False, "cache_page_groups": [group]})))
    allocation = memory.cache_allocation(9)
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [96.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [48.0]
    assert allocation["allocated"]["reserve_bytes_per_stage_rank"] == [0.0]
    assert allocation["valid"]["payload_bytes_per_stage_rank"] == [80.0]
    assert allocation["valid"]["scale_bytes_per_stage_rank"] == [40.0]
    assert allocation["page_tail"]["payload_bytes_per_stage_rank"] == [16.0]
    assert allocation["page_tail"]["scale_bytes_per_stage_rank"] == [8.0]
    assert allocation["allocated"]["metadata_bytes_per_stage_rank"] == [64.0]
    assert allocation["allocated"]["total_bytes_per_stage_rank"] == [208.0]
    assert memory.cache_bytes(9) == (208.0,)


def test_cp2_group_valid_rows_use_local_shard_not_global_sequence():
    raw = case(tp=2)
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    # Use the actual capacity geometry with CP=2; schema head-domain validation
    # is intentionally bypassed because the review fixture declares one KV
    # head/CP2 while cache geometry itself is well-defined.
    parallel = replace(parsed.runtime.parallel, cp=2, dcp=2, cp_plan="dcp")
    plan = replace(parsed.model.layer_plans[0], num_kv_heads=1, head_dim=8)
    model = replace(parsed.model, layers=1, layer_plans=(plan,), metadata=((
        "cache_representations", {
            "gqa_kv": {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"},
        }),))
    runtime = replace(parsed.runtime, parallel=parallel,
                      pp_layer_partition=(1,), memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "layers": [0], "components": ["gqa_kv"],
            "tokens_per_page": 3, "bytes_per_group_page": 72,
        }],
    })
    memory = Memory(replace(parsed, model=model, runtime=runtime))
    # Two global tokens split one local row per CP rank; three physical slots
    # are still allocated, so allocated is 48/24 and valid is 16/8.
    assert memory.cache_bytes(2) == (72.0,)
    assert memory.cache_bytes(2, padded=False) == (24.0,)
    allocation = memory.cache_allocation(2)
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [48.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [24.0]
    assert allocation["valid"]["payload_bytes_per_stage_rank"] == [16.0]
    assert allocation["valid"]["scale_bytes_per_stage_rank"] == [8.0]
    assert allocation["page_tail"]["payload_bytes_per_stage_rank"] == [32.0]
    assert allocation["page_tail"]["scale_bytes_per_stage_rank"] == [16.0]


def test_grouped_page_keeps_known_scale_page_tail_and_access_metadata():
    parsed = compressed_case()
    owner, consumer = parsed.model.layer_plans
    model = replace(
        parsed.model,
        quant=replace(parsed.model.quant, index_dtype="int8",
                      index_scale_dtype="fp16"),
        layer_plans=(
            replace(owner, owns_index_k=True, index_dim=4, index_heads=2,
                    index_input="compressed"),
            replace(consumer, owns_long_kv=True, kv_owner_layer_id=1)))
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "members": [
                {"layer": 0, "component": "compressed_kv"},
                {"layer": 0, "component": "index_k"},
                {"layer": 1, "component": "compressed_kv"},
            ],
            "tokens_per_page": 8, "bytes_per_group_page": 100,
        }],
    })
    memory = Memory(replace(parsed, model=model, runtime=runtime))
    allocation = memory.cache_allocation(9)
    # Two group pages (200 B) contain two int8 index rows with 4 B payload and
    # 2 B scale each. The group price must not erase the physical scale.
    assert allocation["allocated"]["total_bytes_per_stage_rank"] == [712.0]
    assert allocation["allocated"]["payload_bytes_per_stage_rank"] == [584.0]
    assert allocation["allocated"]["scale_bytes_per_stage_rank"] == [4.0]
    assert allocation["allocated"]["reserve_bytes_per_stage_rank"] == [124.0]
    assert allocation["valid"]["total_bytes_per_stage_rank"] == [492.0]
    assert allocation["valid"]["payload_bytes_per_stage_rank"] == [488.0]
    assert allocation["valid"]["scale_bytes_per_stage_rank"] == [4.0]
    assert allocation["page_tail"]["total_bytes_per_stage_rank"] == [220.0]
    assert allocation["page_tail"]["payload_bytes_per_stage_rank"] == [96.0]
    assert allocation["page_tail"]["scale_bytes_per_stage_rank"] == [0.0]
    assert allocation["page_tail"]["reserve_bytes_per_stage_rank"] == [124.0]
    assert allocation["access"]["total_bytes_per_stage_rank"] == [492.0]

    # A 66-page fixed reservation with one touched token allocates all page
    # metadata but reads only the touched page's record.
    runtime_meta = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "layers": [0], "components": ["window"], "tokens_per_page": 128,
            "bytes_per_group_page": 462848, "resident_tokens": 128,
            "max_in_flight_tokens": 8192, "recycling_pages": 1,
            "allocation": "max_model_len", "metadata_bytes_per_page": 32,
        }],
    })
    metadata_memory = Memory(replace(parsed, runtime=runtime_meta))
    group = metadata_memory.pages.groups[0]
    assert group.pages(1, parsed.runtime.max_model_len) == 66
    assert group.touched_pages(1, parsed.runtime.max_model_len) == 1
    metadata = metadata_memory.cache_allocation(1)
    assert metadata["allocated"]["metadata_bytes_per_stage_rank"] == [66 * 32]
    assert metadata["access"]["metadata_bytes_per_stage_rank"] == [32]
    assert metadata["page_tail"]["metadata_bytes_per_stage_rank"] == [65 * 32]

    # An explicit full-scan contract is different from the touched-page view.
    runtime_scan = replace(runtime_meta, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "layers": [0], "components": ["window"], "tokens_per_page": 128,
            "bytes_per_group_page": 462848, "resident_tokens": 128,
            "max_in_flight_tokens": 8192, "recycling_pages": 1,
            "allocation": "max_model_len", "metadata_bytes_per_page": 32,
            "metadata_scan": "all",
        }],
    })
    scanned = Memory(replace(parsed, runtime=runtime_scan)).cache_allocation(1)
    assert scanned["access"]["metadata_bytes_per_stage_rank"] == [66 * 32]
    assert scanned["page_tail"]["metadata_bytes_per_stage_rank"] == [0.0]


def test_joint_hbm_host_grant_is_atomic_across_domains():
    raw = _glm_sparse_case(context_tokens=4096, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload",
        "options": {"dram_size_per_dp_gb": 0.02, "keep_device_kv_cache": False},
    })
    memory = Memory(parse_case(raw))
    # One domain over budget: no HBM or host grant is committed.
    assert not memory.grant_batch(
        0, [("joint:hbm", 8192)], host_bytes=9_000_000)
    assert memory.live_grants(0, 0) == 0.0
    assert memory.granted_host == [0.0]
    assert memory.grant_bindings == {}
    # Both domains fit: commit both; release clears both.
    assert memory.grant_batch(
        0, [("joint:hbm", 4096)], host_bytes=4096)
    assert memory.live_grants(0, 0) == 4096
    assert memory.granted_host == [4096.0]
    assert memory.release_grants(0, ["joint:hbm"]) == 4096
    assert memory.granted_host == [0.0]
    assert memory.grant_bindings == {}


def test_staging_and_grant_share_peak_without_double_counting():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 896)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False, kvc_ddr_ratio=0.5))
    assert memory.capacity == [[896.0]]
    assert memory.resize(0, "r", 8)
    assert memory.live == [[512.0]]
    assert memory.staging_live == [256.0]
    assert memory.hold_for_async_consumer(0, "r", "reader")
    # 512 resident + 256 staging + 256 extra > 896.
    assert memory.grant_batch(0, [("reader-extra", 256)]) is False
    assert memory.live_grants(0, 0) == 0.0
    assert memory.peak_combined[0][0] == 768.0
    # A separately completed run can release both resident and staging, then
    # a standalone 896 grant fits; the historical peak is max(768, 896).
    memory.force_release(0, "r")
    assert memory.grant_batch(0, [("standalone", 896)])
    assert memory.peak_combined[0][0] == 896.0
    assert memory.release_grants(0, ["standalone"]) == 896


def test_kv_offload_fixed_plan_matches_resize_residency_and_staging():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    # Two layers * 8 tokens * 64 B = 1024 B gross; 50% DDR gives 512 B
    # resident + 256 B staging. 896 B free therefore admits the plan.
    capacity = int(base.weights[0] + 896)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False, kvc_ddr_ratio=0.5))
    assert memory.capacity == [[896.0]]
    assert memory.cache_bytes(8) == (1024.0,)
    assert memory._max_layer_cache_bytes(8) == (512.0,)
    assert memory.resize(0, "r", 8) is True
    assert memory.live == [[512.0]]
    # Peak and admission share resident + staging, not gross KV.
    assert memory.peak == [[768.0]]
    assert memory.peak_kv_resident == [[512.0]]
    assert memory.peak_kv_staging == [[256.0]]
    report = memory.allocation_report(8)
    assert report["kv_offload"]["gross_kv_bytes_per_stage_rank"] == [1024.0]
    assert report["kv_offload"]["hbm_resident_kv_bytes_per_stage_rank"] == [512.0]
    assert report["kv_offload"]["staging_bytes_per_stage_rank"] == [256.0]
    assert report["peak_resident_bytes_per_replica_stage"] == [[
        int(base.weights[0] + 768)]]
    assert report["rank_plans"][0][0]["peak_bytes"] == int(base.weights[0] + 768)
    assert report["headroom_bytes_per_replica_stage"] == [[128.0]]
    totals = memory.to_dict()["allocation_totals_at_max_model_len"]
    assert totals["cache_gross_bytes_per_stage_rank"] == [memory.cache_bytes(
        parsed.runtime.max_model_len)[0]]
    assert totals["cache_hbm_resident_bytes_per_stage_rank"][0] + \
        totals["cache_staging_bytes_per_stage_rank"][0] < \
        totals["cache_gross_bytes_per_stage_rank"][0]


def test_async_consumer_holds_keep_cache_until_the_final_real_consumer():
    parsed = parse_case(case())
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "r", 4)
    held = memory.live[0][0]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    assert memory.hold_for_async_consumer(0, "r", "collective")
    assert memory.pending_async_consumers(0, "r") == ("attention", "collective")
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.live == [[held]]
    assert memory.pending_async_consumers(0, "r") == ("collective",)
    # The request is still live: its current consumer completing is not a
    # request retirement and must not free committed KV for the next step.
    assert memory.release_async_consumer(0, "r", "collective") is True
    assert memory.live == [[held]]
    assert memory.owners[(0, "r")] == tuple(memory.cache_bytes(4))
    memory.release(0, "r")
    assert memory.live == [[0.0]]
    assert memory.release_async_consumer(0, "r", "collective") is False


def test_async_hold_prevents_resize_from_reclaiming_physical_backing():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 1024)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory.capacity == [[1024.0]]
    assert memory.resize(0, "r", 8)
    assert memory.live == [[1024.0]]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    # Logical demand shrinks, but the consumer still owns the old backing.
    assert memory.resize(0, "r", 4)
    assert memory.logical_owners[(0, "r")] == (512.0,)
    assert memory.owners[(0, "r")] == (1024.0,)
    assert memory.live == [[1024.0]]
    assert not memory.resize(0, "other", 1)
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.owners[(0, "r")] == (512.0,)
    assert memory.live == [[512.0]]
    assert memory.resize(0, "other", 1)
    assert memory.live == [[640.0]]

    # Two consumers with different registered ranges: the union stays until
    # both complete; completion only reclaims the no-longer-referenced tail.
    memory2 = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory2.resize(0, "r", 8)
    assert memory2.hold_for_async_consumer(0, "r", "first")
    assert memory2.resize(0, "r", 4)
    assert memory2.hold_for_async_consumer(0, "r", "second")
    # The second consumer registered only the shrunken 512 B range; it must
    # not inherit the first consumer's old 1024 B backing.
    assert memory2.release_async_consumer(0, "r", "first") is True
    assert memory2.owners[(0, "r")] == (512.0,)
    assert memory2.live == [[512.0]]
    assert memory2.release_async_consumer(0, "r", "second") is True
    assert memory2.owners[(0, "r")] == (512.0,)
    assert memory2.live == [[512.0]]
def test_sparse_host_dram_shrink_uses_the_same_hold_union_and_admission():
    raw = _glm_sparse_case(context_tokens=4096, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload",
        "options": {"dram_size_per_dp_gb": 0.02, "keep_device_kv_cache": False},
    })
    memory = Memory(parse_case(raw))
    assert memory.host_dram_capacity == [8891924.0]
    assert memory.host_dram_cache_bytes(2048) == 4718592.0
    assert memory.host_dram_cache_bytes(1024) == 2359296.0
    # Empty ledger, same budget as resize: max span is 3840, not 4096.
    assert memory.admissible_tokens(0, "r", 4096, granularity=128) == 3840
    assert memory.live == [[0.0]]
    assert memory.host_dram_live == [0.0]
    assert memory.owners == {}
    assert memory.resize(0, "r", 2048)
    assert memory.host_dram_live == [4718592.0]
    assert memory.hold_for_async_consumer(0, "r", "consumer")
    assert memory.resize(0, "r", 1024)
    assert memory.logical_host_owners[(0, "r")] == 2359296.0
    assert memory.host_dram_owners[(0, "r")] == 4718592.0
    assert memory.host_dram_live == [4718592.0]
    # 4,718,592 + 4,718,592 = 9,437,184 > 8,891,924.  A failed resize must
    # not commit HBM or host changes partially.
    before_live = [list(row) for row in memory.live]
    before_host = list(memory.host_dram_live)
    assert not memory.resize(0, "other", 2048)
    assert memory.live == before_live
    assert memory.host_dram_live == before_host
    assert memory.release_async_consumer(0, "r", "consumer")
    assert memory.host_dram_live == [2359296.0]
    assert memory.resize(0, "other", 2048)
    assert memory.host_dram_live == [7077888.0]


def test_retire_removes_own_demand_and_keeps_only_other_consumer_ranges():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 1024)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "r", 4)
    assert memory.hold_for_async_consumer(0, "r", "A")
    assert memory.resize(0, "r", 8)
    assert memory.hold_for_async_consumer(0, "r", "B")
    memory.release(0, "r")
    assert memory.is_retired(0, "r")
    assert memory.logical_owners[(0, "r")] == (0.0,)
    # B still owns its 1024 B range, so another request cannot yet allocate.
    assert not memory.resize(0, "other", 4)
    assert memory.release_async_consumer(0, "r", "B")
    assert memory.owners[(0, "r")] == (512.0,)
    assert memory.live == [[512.0]]
    # A's 512 B remains live; a second 512 B request fits in the 1024 B budget.
    assert memory.resize(0, "other", 4)
    assert memory.live == [[1024.0]]
    assert memory.release_async_consumer(0, "r", "A")
    assert memory.live == [[512.0]]


def test_async_retired_request_waits_for_pending_consumer_and_delayed_expiry():
    parsed = parse_case(case())
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "r", 4)
    held = memory.live[0][0]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    # Retire before the consumer's completion. The owner remains resident.
    memory.release(0, "r")
    assert memory.is_retired(0, "r")
    assert memory.live == [[held]]
    assert memory.pending_async_consumers(0, "r") == ("attention",)
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.live == [[0.0]]
    assert not memory.is_retired(0, "r")

    # A delayed lookahead expiry is also not a force-retire: pending holds stay.
    deferred = Memory(parsed, MemoryPolicy(
        derive_workspace=False, async_delayed_release_steps=1))
    assert deferred.resize(0, "r", 4)
    assert deferred.hold_for_async_consumer(0, "r", "attention")
    deferred.defer_release(0, "r", 1)
    deferred.step_delayed_releases(0)
    assert deferred.is_retired(0, "r")
    assert deferred.live == [[held]]
    assert deferred.pending_async_consumers(0, "r") == ("attention",)
    assert deferred.delayed_owners == {}
    assert deferred.release_async_consumer(0, "r", "attention") is True
    assert deferred.live == [[0.0]]

    # Cancellation/rollback has an explicit force path.
    forced = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert forced.resize(0, "r", 4)
    assert forced.hold_for_async_consumer(0, "r", "attention")
    forced.force_release(0, "r")
    assert forced.live == [[0.0]]
    assert forced.pending_async_consumers(0, "r") == ()


def test_fixed_plan_over_capacity_does_not_switch_dtype_or_algorithm():
    parsed = parse_case(case())
    base = Memory(parsed, MemoryPolicy(derive_workspace=False))
    tight = replace(base.case, machine=replace(
        base.case.machine,
        dies=tuple(replace(die, memory_bytes=int(base.weights[0] - 1))
                   for die in base.case.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    report = memory.to_dict()
    assert report["status"] == "oom"
    assert report["fixed_plan"] is True and report["fallback"] == "none"
    assert memory.resize(0, "r", 1) is False
    assert memory.weights == base.weights
    assert memory.case.model.quant == tight.model.quant


def test_real_a5_capacity_report_explains_payload_tail_scale_and_metadata():
    qwen = Memory(parse_case(json.loads(A5_QWEN.read_text())))
    glm = Memory(parse_case(json.loads(A5_GLM.read_text())))
    v41 = Memory(parse_case(json.loads(A5_V41.read_text())))
    for memory, expected_tail in ((qwen, 0.0), (glm, 2883584.0)):
        span = memory.case.runtime.max_model_len
        report = memory.allocation_report(span)
        assert report["fixed_plan"] is True and report["fallback"] == "none"
        assert report["weights"]["total_bytes_per_stage_rank"] == memory.weights
        assert report["weights"]["payload_bytes_per_stage_rank"] == memory.weights_payload
        assert [p + m for p, m in zip(
            report["weights"]["payload_bytes_per_stage_rank"],
            report["weights"]["quant_metadata_bytes_per_stage_rank"])] == memory.weights
        assert report["cache"]["allocated"]["total_bytes_per_stage_rank"] == list(
            memory.cache_bytes(span))
        assert report["cache"]["page_tail"]["total_bytes_per_stage_rank"] == [expected_tail]
        assert report["cache"]["valid"]["total_bytes_per_stage_rank"] == [
            payload + scale for payload, scale in zip(
                report["cache"]["valid"]["payload_bytes_per_stage_rank"],
                report["cache"]["valid"]["scale_bytes_per_stage_rank"])]
        assert all(value >= 0.0 for row in report["headroom_bytes_per_replica_stage"]
                   for value in row)
        # Real A5 deployment, not an abstract hardware label.
        assert "a5-" in memory.case.machine.name
    # The W8A8-MXFP8 A5 model publishes non-zero weight scale metadata; the
    # BF16 GLM checkpoint does not invent any.
    assert qwen.weights_quant_metadata[0] > 0.0
    assert glm.weights_quant_metadata[0] == 0.0
    # Page-tail is allocated capacity, not a read: access is the valid view.
    assert glm.cache_allocation(glm.case.runtime.max_model_len)["access"][
        "total_bytes_per_stage_rank"][0] < sum(glm.cache_bytes(
            glm.case.runtime.max_model_len))
    # The real A5 V4.1 compressed cache declares row scales; they are a
    # separate allocation component rather than being folded into payload.
    v41_alloc = v41.cache_allocation(v41.case.runtime.max_model_len)
    assert v41_alloc["allocated"]["scale_bytes_per_stage_rank"][0] > 0.0
    assert v41_alloc["access"]["scale_bytes_per_stage_rank"][0] > 0.0
    assert [payload + scale + metadata for payload, scale, metadata in zip(
        v41_alloc["allocated"]["payload_bytes_per_stage_rank"],
        v41_alloc["allocated"]["scale_bytes_per_stage_rank"],
        v41_alloc["allocated"]["metadata_bytes_per_stage_rank"])] == [
            *v41_alloc["allocated"]["total_bytes_per_stage_rank"]]
    # A real public rank plan carries the same physical categories.
    plan = qwen.rank_capacity_plan(0, 0, tokens=qwen.case.runtime.max_model_len)
    resolved = plan.resolve()
    assert resolved["feasible"] is True
    assert resolved["categories"]["cache"] == qwen.cache_bytes(
        qwen.case.runtime.max_model_len)[0]

def test_per_item_grant_bindings_and_generation_safe_release():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 4 * 4096)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    issue = object()
    grants = [
        {"name": "src", "bytes": 4096, "owner": "producer:src",
         "storage": "K", "tier": "hbm", "rank": 1, "stage": 0,
         "generation": 3, "ranges": (0, 4096), "host_bytes": 512,
         "event": issue, "physical_key": "K:page1"},
        {"name": "dst", "bytes": 4096, "owner": "consumer:dst",
         "storage": "V", "tier": "hbm", "rank": 1, "stage": 0,
         "generation": 4, "ranges": (8192, 12288), "host_bytes": 0,
         "event": issue, "physical_key": "V:page1"},
    ]
    version = memory.budget_version
    assert memory.grant_batch(0, []) is True
    assert memory.budget_version == version
    assert memory.grant_batch(0, grants, stage=0)
    assert memory.budget_version > version
    assert memory.grant_bindings[(0, "src")]["storage"] == "K"
    assert memory.grant_bindings[(0, "src")]["generation"] == 3
    assert memory.grant_bindings[(0, "src")]["ranges"] == (0, 4096)
    assert memory.grant_bindings[(0, "dst")]["storage"] == "V"
    assert memory.grant_bindings[(0, "dst")]["generation"] == 4
    assert memory.live_grants(0, 0) == 8192
    # Wrong generation and duplicate names cannot release or alter a holder.
    held_version = memory.budget_version
    assert memory.release_grant(0, "src", generation=4) is False
    assert memory.release_grant(0, "src", event=object()) is False
    assert memory.budget_version == held_version
    assert memory.live_grants(0, 0) == 8192
    assert memory.grant_batch(0, [("src", 4096)]) is False
    assert memory.budget_version == held_version
    assert memory.release_grant(0, "src", generation=3) is True
    assert memory.live_grants(0, 0) == 4096
    holders = memory.outstanding_holders()
    assert holders["budget_version"] > version
    assert [row["name"] for row in holders["grants"]] == ["dst"]
    released = memory.drain_release_events()
    assert released and released[-1]["kind"] == "grant"
    assert released[-1]["generation"] == 3
    assert memory.release_grant(0, "dst", generation=4) is True
    assert memory.live_grants(0, 0) == 0.0
    # Legacy uniform metadata still counts distinct names independently.
    assert memory.grant_batch(0, [("u1", 4096), ("u2", 4096)],
                              owner="same-task")
    assert memory.live_grants(0, 0) == 8192
    assert memory.release_grants(0, ["u1", "u2"]) == 8192
    assert memory.live_grants(0, 0) == 0.0


def test_selected_acquires_are_incremental_and_grant_duplicates_reject():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 12288)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("acquire:a", 4096, "workspace"),)))
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("acquire:b", 4096, "workspace"),)))
    assert memory.selected_reserved == [[8192.0]]
    assert memory.grant_batch(0, [("grant:8k", 8192)]) is False
    assert memory.selected_reserved == [[8192.0]]
    assert memory.granted == [[0.0]]
    assert memory.release_selected_allocation("acquire:a")
    assert memory.selected_reserved == [[4096.0]]
    alias_plan = AllocationPlan((
        Allocation("alias:backing", 4096, "workspace"),
        Allocation("alias:view", 4096, "workspace",
                   alias_of="alias:backing"),
    ))
    assert memory.attach_allocation_plan(alias_plan, replace=True)
    assert memory.selected_reserved == [[4096.0]]
    assert memory.release_selected_allocation("alias:backing")
    assert memory.selected_reserved == [[4096.0]]
    assert memory.release_selected_allocation("alias:view")
    assert memory.selected_reserved == [[0.0]]
    assert memory.grant_batch(0, [("grant:8k", 8192)]) is True
    assert memory.live_grants(0, 0) == 8192
    assert memory.release_grants(0, ["grant:8k"]) == 8192
    assert memory.live_grants(0, 0) == 0.0
    assert memory.peak_combined[0][0] >= 8192.0


def test_selected_allocations_and_shared_pages_use_the_same_memory_budget():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 8192)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    selected = AllocationPlan((
        Allocation("selected:workspace", 4096, "workspace"),
        Allocation("selected:alias", 4096, "workspace",
                   alias_of="selected:workspace"),
    ))
    assert memory.attach_allocation_plan(selected)
    assert memory.selected_reserved == [[4096.0]]
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("too-large", 8192, "workspace"),))) is False
    assert memory.release_selected_plan()
    assert memory.selected_reserved == [[0.0]]
    assert memory.attach_allocation_effects((
        AllocationEffect.acquire("workspace", "L2", 4096, unit="bytes"),
    ))
    # On-chip L2 remains T04's pool and adds zero HBM bytes.
    assert memory.selected_reserved == [[0.0]]
    assert memory.onchip_effects
    assert memory.grant_batch(0, [("hbm:8k", 8192)])
    assert memory.release_grants(0, ["hbm:8k"]) == 8192

    # Two readers of one physical prefix owner charge the range once.
    assert memory.retain_shared_pages(0, "prefix:page-owner", (4096.0,))
    assert memory.retain_shared_pages(0, "prefix:page-owner", (4096.0,))
    assert memory.shared_reserved == [[4096.0]]
    assert memory.shared_pages[(0, "prefix:page-owner")]["refs"] == 2
    assert memory.release_shared_pages(0, "prefix:page-owner")
    assert memory.shared_reserved == [[4096.0]]
    assert memory.release_shared_pages(0, "prefix:page-owner")
    assert memory.shared_reserved == [[0.0]]
    assert memory.shared_pages == {}

    # Two request owners of the same physical page are charged once.
    assert memory.retain_shared_pages(0, "request-a", (4096.0,),
                                      physical_key="physical-page")
    assert memory.retain_shared_pages(0, "request-b", (4096.0,),
                                      physical_key="physical-page")
    assert memory.shared_reserved == [[4096.0]]
    assert memory.shared_pages[(0, "physical-page")]["refs"] == 2
    assert memory.release_shared_pages(0, "request-a")
    assert memory.shared_reserved == [[4096.0]]
    assert memory.release_shared_pages(0, "request-b")
    assert memory.shared_reserved == [[0.0]]

    # A selected backing and a request shared page with the same physical key
    # are one physical allocation; releasing selected transfers charge to the
    # remaining reader rather than double counting or dropping it.
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("selected:physical", 4096, "workspace", physical_key="page:P"),)))
    assert memory.selected_reserved == [[4096.0]]
    assert memory.retain_shared_pages(
        0, "reader", (4096.0,), physical_key="page:P")
    assert memory.selected_reserved == [[4096.0]]
    assert memory.shared_reserved == [[0.0]]
    assert memory.release_selected_allocation("selected:physical")
    assert memory.selected_reserved == [[0.0]]
    assert memory.shared_reserved == [[4096.0]]
    assert memory.release_shared_pages(0, "reader")
    assert memory.shared_reserved == [[0.0]]


def test_batch_grant_is_all_or_nothing_against_the_same_budget():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 3 * 4096)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory.grant_batch(0, [("a", 4096), ("b", 4096), ("c", 4096)])
    assert memory.live_grants(0, 0) == 12288
    assert memory.grant_batch(0, [("a", 4096)]) is False
    assert memory.live_grants(0, 0) == 12288
    assert memory.grant_batch(0, [("d", 4096)]) is False
    assert memory.live_grants(0, 0) == 12288
    assert memory.release_grants(0, ["b"]) == 4096
    assert memory.live_grants(0, 0) == 8192
    assert memory.grant_batch(0, [("d", 4096)])
    assert memory.live_grants(0, 0) == 12288
    # A batch with one oversized member changes nothing.
    assert memory.grant_batch(0, [("e", 4096), ("f", 8192)]) is False
    assert memory.live_grants(0, 0) == 12288


def test_admissible_tokens_is_a_query_and_does_not_mutate_the_ledger():
    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 1024)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    assert memory.admissible_tokens(0, "r", 16) == 8
    assert memory.admissible_tokens(0, "r", 16, granularity=2) == 8
    assert memory.admissible_tokens(0, "r", 7) == 7
    assert memory.live == [[0.0]]
    assert memory.owners == {}
    assert memory.resize(0, "r", 8)


def test_physical_rank_capacity_plans_do_not_collapse_heterogeneous_dies():
    raw = case(tp=2)
    raw["runtime"].update(memory={"derive_workspace": False},
                          gpu_memory_utilization=1.0, kv_block_size=1)
    parsed = parse_case(raw)
    weights = Memory(parsed).weights[0]
    machine = replace(parsed.machine, dies=(
        replace(parsed.machine.dies[0], memory_bytes=int(weights + 1000)),
        replace(parsed.machine.dies[1], memory_bytes=int(weights + 2000)),
    ))
    memory = Memory(replace(parsed, machine=machine),
                    MemoryPolicy(derive_workspace=False))
    assert memory.capacity == [[1000.0]]
    mounted = [die.hbm_bytes for die in memory._stage_dies(0, 0)]
    plans = memory.rank_capacity_plans(0, 0, tokens=1)
    assert [plan.capacity_bytes for plan in plans] == mounted
    assert plans[0].resolve()["feasible"] is True
    assert plans[1].resolve()["feasible"] is True
    assert memory.rank_capacity_plan(0, 0, tokens=1, rank=0).capacity_bytes == mounted[0]
    report = memory.to_dict()
    assert report["physical_rank_capacity_bytes_per_replica_stage"] == [[mounted]]


def test_real_scheduler_execution_share_the_same_memory_grant_budget():
    raw = case(requests=[{"id": "r", "prompt_tokens": 4, "output_tokens": 1}])
    raw["runtime"]["gpu_memory_utilization"] = 1.0
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + base.workspace[0]
                   + base.cache_bytes(4)[0] + 3 * 4096)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight)
    assert memory.capacity == [[3 * 4096 + base.cache_bytes(4)[0]]]
    scheduler = Scheduler(tight, list(tight.workload.requests), memory)
    plan = scheduler.schedule(0)
    assert plan is not None
    work = linear_work(plan, tight.model, tight.runtime, tight.machine)
    execution = Execution(tight, CostModel(tight), include_submission=False)
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,))
    assert step.completion.done

    assert memory.grant_batch(0, [
        ("exec:attn", 4096), ("exec:workspace", 4096)])
    assert memory.grant_batch(0, [("exec:extra", 4096)])
    assert memory.live_grants(0, 0) == 12288
    assert memory.grant_batch(0, [("exec:fourth", 4096)]) is False
    assert memory.live_grants(0, 0) == 12288
    assert memory.hold_for_async_consumer(
        0, "r", "exec:last-reader", storage="exec:workspace",
        ranges=(4096,), generation=1)
    assert memory.release_grants(0, ["exec:workspace"]) == 4096
    assert memory.grant_batch(0, [("exec:fourth", 4096)])
    assert memory.live_grants(0, 0) == 12288
    assert memory.release_async_consumer(0, "r", "exec:last-reader")
    memory.release(0, "r")
    assert memory.live == [[0.0]]
    assert memory.live_grants(0, 0) == 12288


def _capacity_memory(budgets, *, tp=1):
    """Normal Case parse with an exact per-die HBM budget after fixed weights."""
    raw = case(tp=tp)
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed, MemoryPolicy(derive_workspace=False))
    if isinstance(budgets, (int, float)):
        budgets = (budgets,) * len(parsed.machine.dies)
    dies = tuple(replace(die, memory_bytes=int(base.weights[0] + budget))
                 for die, budget in zip(parsed.machine.dies, budgets))
    return Memory(replace(parsed, machine=replace(parsed.machine, dies=dies)),
                  MemoryPolicy(derive_workspace=False))


def _cp2_owner_case(*, page_group):
    raw = case(tp=2)
    raw["model"].update(
        layers=1, num_kv_heads=1, head_dim=8,
        metadata={"cache_representations": {
            "gqa_kv": {"dtype": "int8", "group_size": 8,
                       "scale_dtype": "fp32"}}})
    raw["runtime"].update(
        tp=2, cp=2, dcp=2, cp_plan="dcp", kv_block_size=1, max_model_len=8,
        gpu_memory_utilization=1.0, memory={"derive_workspace": False})
    if page_group:
        raw["runtime"]["memory"]["cache_page_groups"] = [{
            "layers": [0], "components": ["gqa_kv"], "tokens_per_page": 3,
            "bytes_per_group_page": 72,
        }]
    return parse_case(raw)


def _cp2_owner_memory(budgets, *, page_group):
    parsed = _cp2_owner_case(page_group=page_group)
    base = Memory(parsed)
    dies = tuple(replace(die, memory_bytes=int(base.weights[0] + budget))
                 for die, budget in zip(parsed.machine.dies, budgets))
    values = dict(parsed.runtime.memory) if page_group else {
        "derive_workspace": False}
    policy = MemoryPolicy(**values, rank_token_owners=(2, 1))
    return Memory(replace(parsed, machine=replace(parsed.machine, dies=dies)),
                  policy)


def test_grant_batch_physical_union_is_atomic_per_rank_and_host():
    event_k, event_v = object(), object()
    grants = [
        {"name": "K", "bytes": 4096, "physical_key": "P", "storage": "K",
         "ranges": (0, 4096), "generation": 3, "event": event_k},
        {"name": "V", "bytes": 4096, "physical_key": "Q", "storage": "V",
         "ranges": (0, 4096), "generation": 4, "event": event_v},
    ]
    tight = _capacity_memory(4096)
    assert tight.capacity == [[4096.0]]
    before = (tight.live, tight.peak_combined, dict(tight.grant_bindings),
              tight.budget_version, tight.occupied_bytes(0, 0),
              tight.host_occupied_bytes(0))
    assert tight.grant_batch(0, grants) is False
    after = (tight.live, tight.peak_combined, dict(tight.grant_bindings),
             tight.budget_version, tight.occupied_bytes(0, 0),
             tight.host_occupied_bytes(0))
    assert after == before
    assert tight.release_events == []

    roomy = _capacity_memory(8192)
    assert roomy.grant_batch(0, grants) is True
    assert roomy.occupied_bytes(0, 0) == 8192.0
    assert roomy.release_grants(0, ["K", "V"]) == 8192.0
    assert roomy.occupied_bytes(0, 0) == 0.0

    host_grants = [
        {"name": "HK", "bytes": 0, "physical_key": "HP", "generation": 3,
         "host_bytes": 4096},
        {"name": "HV", "bytes": 0, "physical_key": "HQ", "generation": 4,
         "host_bytes": 4096},
    ]
    host = _capacity_memory(8192)
    host.host_dram_capacity = [8191.0]
    assert host.grant_batch(0, host_grants) is False
    assert host.host_occupied_bytes(0) == 0.0
    assert host.grant_bindings == {}
    host.host_dram_capacity = [8192.0]
    assert host.grant_batch(0, host_grants) is True
    assert host.host_occupied_bytes(0) == 8192.0

    ranks = _capacity_memory([4096, 4096], tp=2)
    split = [
        {"name": "K", "bytes": 4096, "physical_key": "P", "storage": "K",
         "ranges": (0, 4096), "generation": 3, "rank": 0},
        {"name": "V", "bytes": 4096, "physical_key": "Q", "storage": "V",
         "ranges": (0, 4096), "generation": 4, "rank": 1},
    ]
    assert ranks.grant_batch(0, split) is True
    assert ranks.occupied_bytes(0, 0, rank=0) == 4096.0
    assert ranks.occupied_bytes(0, 0, rank=1) == 4096.0


def test_same_backing_holders_ranges_generations_and_selected_shared_orders():
    def grant(name, ranges, generation, key="P"):
        return {"name": name, "bytes": 4096, "physical_key": key,
                "storage": "S", "ranges": ranges, "generation": generation}

    memory = _capacity_memory(8192)
    assert memory.grant_batch(0, [grant("A", (0, 4096), 3),
                                  grant("B", (0, 4096), 3)])
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.peak_combined[0][0] == 4096.0
    assert memory.release_grant(0, "A", generation=3) is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_grant(0, "B", generation=3) is True
    assert memory.occupied_bytes(0, 0) == 0.0

    memory = _capacity_memory(8192)
    assert memory.grant_batch(0, [grant("A", (0, 4096), 3)])
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.grant_batch(0, [grant("B", (2048, 6144), 3)])
    assert memory.occupied_bytes(0, 0) == 6144.0
    assert memory.release_grant(0, "A", generation=3) is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_grant(0, "B", generation=3) is True
    assert memory.occupied_bytes(0, 0) == 0.0

    memory = _capacity_memory(8192)
    assert memory.grant_batch(0, [grant("A", (0, 4096), 3),
                                  grant("B", (0, 4096), 4)])
    # Concurrent live generations are different allocations, never merged.
    assert memory.occupied_bytes(0, 0) == 8192.0

    for selected_first in (True, False):
        memory = _capacity_memory(4096)
        selected = AllocationPlan((
            Allocation("selected:P", 4096, "workspace",
                       physical_key="P", storage="S"),))
        if not selected_first:
            assert memory.retain_shared_pages(
                0, "reader", (4096.0,), physical_key="P", storage="S")
        assert memory.attach_allocation_plan(selected)
        if selected_first:
            assert memory.retain_shared_pages(
                0, "reader", (4096.0,), physical_key="P", storage="S")
        assert memory.occupied_bytes(0, 0) == 4096.0
        assert memory.peak_combined[0][0] == 4096.0
        assert memory.selected_reserved == [[4096.0]]
        assert memory.shared_reserved == [[0.0]]
        assert memory.release_selected_allocation("selected:P") is True
        assert memory.occupied_bytes(0, 0) == 4096.0
        assert memory.selected_reserved == [[0.0]]
        assert memory.shared_reserved == [[4096.0]]
        assert memory.release_shared_pages(0, "reader") is True
        assert memory.occupied_bytes(0, 0) == 0.0
        assert memory.peak_combined[0][0] == 4096.0

    memory = _capacity_memory(4096)
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("selected:P", 4096, "workspace",
                   physical_key="P", storage="S"),)))
    assert memory.retain_shared_pages(
        0, "old-reader", (4096.0,), physical_key="P", storage="S")
    version = memory.budget_version
    assert memory.attach_allocation_plan(AllocationPlan((
        Allocation("selected:Q", 4096, "workspace",
                   physical_key="Q", storage="S"),)), replace=True) is False
    assert memory.budget_version == version
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.peak_combined[0][0] == 4096.0
    assert list(memory.selected_items[(0, 0)]) == ["selected:P"]


def test_grant_reader_last_use_keeps_physical_backing_until_completion():
    memory = _capacity_memory(12288)
    events = [object(), object(), object(), object()]
    assert memory.grant_batch(0, [
        {"name": "L0", "bytes": 4096, "physical_key": "P0",
         "storage": "L0", "ranges": (0, 4096), "generation": 3,
         "event": events[0]},
        {"name": "L1", "bytes": 4096, "physical_key": "P1",
         "storage": "L1", "ranges": (0, 4096), "generation": 4,
         "event": events[1]},
        {"name": "L2", "bytes": 4096, "physical_key": "P2",
         "storage": "L2", "ranges": (0, 4096), "generation": 5,
         "event": events[2]},
    ])
    assert memory.occupied_bytes(0, 0) == 12288.0
    assert memory.hold_for_async_consumer(
        0, "L0", "reader:L0", grant="L0", generation=3,
        event=events[0]) is True
    assert memory.release_grant(
        0, "L0", generation=3, event=events[0]) is True
    # The grant handle is logically withdrawn but its reader still owns 4 KiB.
    assert memory.occupied_bytes(0, 0) == 12288.0
    assert memory.live_grants(0, 0) == 8192.0
    l3 = {"name": "L3", "bytes": 4096, "physical_key": "P3",
          "storage": "L3", "ranges": (0, 4096), "generation": 6,
          "event": events[3]}
    version = memory.budget_version
    assert memory.grant_batch(0, [l3]) is False
    assert memory.budget_version == version
    assert memory.occupied_bytes(0, 0) == 12288.0
    assert memory.release_grant(0, "L0", generation=4) is False
    assert memory.release_async_consumer(0, "L0", "other") is False
    assert memory.release_async_consumer(
        0, "L0", "reader:L0", generation=4) is False
    assert memory.release_async_consumer(
        0, "L0", "reader:L0", event=object()) is False
    assert memory.budget_version == version
    assert memory.occupied_bytes(0, 0) == 12288.0
    assert memory.release_async_consumer(
        0, "L0", "reader:L0", generation=3, event=events[0]) is True
    assert memory.occupied_bytes(0, 0) == 8192.0
    released = memory.drain_release_events()
    assert [(row["kind"], row["released_bytes"]) for row in released] == [
        ("grant_reader", 4096.0)]
    assert memory.grant_batch(0, [l3]) is True
    assert memory.occupied_bytes(0, 0) == 12288.0
    assert memory.release_grants(0, ["L1", "L2", "L3"]) == 12288.0
    assert memory.occupied_bytes(0, 0) == 0.0
    assert {row["kind"] for row in memory.drain_release_events()} == {"grant"}


def test_uneven_token_owners_drive_per_rank_geometry_and_admission():
    grouped = _cp2_owner_memory((1000, 1000), page_group=True)
    views = grouped.rank_cache_allocation(3)
    assert grouped.cache_bytes(3) == (72.0,)
    assert views[0]["allocated"]["payload_bytes_per_stage_rank"] == [48.0]
    assert views[0]["allocated"]["scale_bytes_per_stage_rank"] == [24.0]
    assert views[1]["allocated"]["payload_bytes_per_stage_rank"] == [48.0]
    assert views[1]["allocated"]["scale_bytes_per_stage_rank"] == [24.0]
    assert views[0]["valid"]["payload_bytes_per_stage_rank"] == [32.0]
    assert views[0]["valid"]["scale_bytes_per_stage_rank"] == [16.0]
    assert views[1]["valid"]["payload_bytes_per_stage_rank"] == [16.0]
    assert views[1]["valid"]["scale_bytes_per_stage_rank"] == [8.0]
    assert views[0]["page_tail"]["payload_bytes_per_stage_rank"] == [16.0]
    assert views[0]["page_tail"]["scale_bytes_per_stage_rank"] == [8.0]
    assert views[1]["page_tail"]["payload_bytes_per_stage_rank"] == [32.0]
    assert views[1]["page_tail"]["scale_bytes_per_stage_rank"] == [16.0]

    exact = _cp2_owner_memory((48, 24), page_group=False)
    assert exact.rank_capacity[0][0] == [48.0, 24.0]
    assert exact.capacity == [[24.0]]
    assert exact.admissible_tokens(0, "r", 4) == 3
    exact_views = exact.rank_cache_allocation(3)
    assert exact_views[0]["allocated"]["payload_bytes_per_stage_rank"] == [32.0]
    assert exact_views[0]["allocated"]["scale_bytes_per_stage_rank"] == [16.0]
    assert exact_views[1]["allocated"]["payload_bytes_per_stage_rank"] == [16.0]
    assert exact_views[1]["allocated"]["scale_bytes_per_stage_rank"] == [8.0]
    assert exact.resize(0, "r", 3) is True
    assert exact.occupied_bytes(0, 0, rank=0) == 48.0
    assert exact.occupied_bytes(0, 0, rank=1) == 24.0
    assert exact.live_rank[0][0] == [48.0, 24.0]
    plans = exact.rank_capacity_plans(0, 0, tokens=3)
    assert [plan.resolve()["categories"]["cache"] for plan in plans] == [
        48.0, 24.0]

    tight_rank1 = _cp2_owner_memory((48, 23), page_group=False)
    before = (tight_rank1.occupied_bytes(0, 0), tight_rank1.live,
              tight_rank1.peak_combined, tight_rank1.budget_version,
              dict(tight_rank1.owners))
    assert tight_rank1.resize(0, "r", 3) is False
    assert (tight_rank1.occupied_bytes(0, 0), tight_rank1.live,
            tight_rank1.peak_combined, tight_rank1.budget_version,
            dict(tight_rank1.owners)) == before
    assert tight_rank1.admissible_tokens(0, "r", 4) == 2

    tight_rank0 = _cp2_owner_memory((47, 24), page_group=False)
    assert tight_rank0.resize(0, "r", 3) is False
    assert tight_rank0.admissible_tokens(0, "r", 4) == 1

    # One rank's own budget rejects even when the other rank is empty.
    one_sided = _cp2_owner_memory((10 ** 9, 23), page_group=False)
    assert one_sided.resize(0, "r", 3) is False
    assert one_sided.occupied_bytes(0, 0, rank=0) == 0.0


def test_r7_finite_regressions_pp2_host_peak_and_retired_tail():
    memory = _capacity_memory(1024)
    assert memory.resize(0, "r", 4) is True
    assert memory.hold_for_async_consumer(0, "r", "A") is True
    assert memory.resize(0, "r", 8) is True
    memory.release(0, "r")
    # Retirement immediately drops the unreferenced old tail to A's 512 B.
    assert memory.occupied_bytes(0, 0) == 512.0
    assert memory.resize(0, "other", 4) is True
    assert memory.occupied_bytes(0, 0) == 1024.0
    assert memory.release_async_consumer(0, "r", "A") is True
    assert memory.occupied_bytes(0, 0) == 512.0

    raw = case(pp=2)
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    dies = tuple(replace(die, memory_bytes=int(weight + 512))
                 for die, weight in zip(parsed.machine.dies, base.weights))
    pp2 = Memory(replace(parsed, machine=replace(parsed.machine, dies=dies)),
                 MemoryPolicy(derive_workspace=False, kvc_ddr_ratio=0.5))
    assert pp2.capacity == [[512.0, 512.0]]
    assert pp2.resize(0, "r", 8) is True
    assert pp2.live == [[256.0, 256.0]]
    assert pp2.staging_rank[0] == [[256.0], [256.0]]
    assert pp2.peak_combined == [[512.0, 512.0]]
    assert pp2.grant_batch(0, [("stage1-extra", 256)], stage=1) is False
    assert pp2.occupied_bytes(0, 1) == 512.0

    raw = _glm_sparse_case(context_tokens=4096, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload",
        "options": {"dram_size_per_dp_gb": 0.02,
                    "keep_device_kv_cache": False},
    })
    host = Memory(parse_case(raw))
    assert host.grant_batch(0, [("h", 0)], host_bytes=3_000_000) is True
    assert host.resize(0, "r", 2048) is True
    assert host.resize(0, "r", 2048) is True
    assert host.host_occupied_bytes(0) == 7_718_592.0
    host.release_grants(0, ["h"])
    assert host.host_occupied_bytes(0) == 4_718_592.0
    host.release(0, "r")
    assert host.host_occupied_bytes(0) == 0.0

    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=int(base.weights[0] + 896))
                   for die in parsed.machine.dies)))
    shared = Memory(tight, MemoryPolicy(derive_workspace=False,
                                        kvc_ddr_ratio=0.5))
    assert shared.resize(0, "r", 8) is True
    assert shared.peak_combined == [[768.0]]
    shared.force_release(0, "r")
    assert shared.grant_batch(0, [("standalone", 896)]) is True
    assert shared.peak_combined == [[896.0]]
    assert shared.to_dict()["peak_resident_bytes_per_replica_stage"] == [
        [int(base.weights[0] + 896)]]


def test_request_admission_decision_names_waitable_and_permanent_shortfall():
    memory = _capacity_memory(1024)
    assert memory.resize(0, "r", 4) is True
    assert memory.occupied_bytes(0, 0) == 512.0
    version = memory.budget_version
    wait = memory.admission_decision(0, "other", 8)
    assert wait["ok"] is False
    assert wait["wakeable"] is True and wait["permanent"] is False
    assert wait["reason"] == "waitable_release"
    assert wait["shortfall_bytes_per_stage_rank"] == [[512.0]]
    assert wait["releasable_bytes_per_stage_rank"] == [[512.0]]
    assert wait["budget_version"] == version
    assert memory.occupied_bytes(0, 0) == 512.0
    permanent = memory.admission_decision(0, "other", 16)
    assert permanent["ok"] is False
    assert permanent["wakeable"] is False and permanent["permanent"] is True
    assert permanent["reason"] == "permanent_budget"
    assert permanent["shortfall_bytes_per_stage_rank"] == [[1536.0]]
    assert memory.budget_version == version
    memory.release(0, "r")
    assert memory.admission_decision(0, "other", 8)["ok"] is True
    assert memory.resize(0, "other", 8) is True
    assert memory.occupied_bytes(0, 0) == 1024.0


def test_grant_decision_names_waitable_release_and_permanent_deficit():
    memory = _capacity_memory(4096)
    event = object()
    version = memory.budget_version
    big = memory.grant_decision(0, [{
        "name": "big", "bytes": 8192, "physical_key": "P", "storage": "S",
        "ranges": (0, 8192), "generation": 1}])
    assert big["ok"] is False
    assert big["reason"] == "permanent_budget"
    assert big["wakeable"] is False and big["permanent"] is True
    assert big["budget_version"] == version
    assert big["shortfall_bytes_per_stage_rank"] == [[4096.0]]
    assert memory.occupied_bytes(0, 0) == 0.0
    assert memory.grant_bindings == {}
    assert memory.release_events == []

    assert memory.grant_batch(0, [{
        "name": "A", "bytes": 4096, "physical_key": "P", "storage": "S",
        "ranges": (0, 4096), "generation": 1, "event": event}]) is True
    held = memory.budget_version
    wait = memory.grant_decision(0, [{
        "name": "B", "bytes": 4096, "physical_key": "Q", "storage": "S",
        "ranges": (0, 4096), "generation": 2}])
    assert wait["ok"] is False
    assert wait["reason"] == "waitable_release"
    assert wait["wakeable"] is True and wait["permanent"] is False
    assert wait["shortfall_bytes_per_stage_rank"] == [[4096.0]]
    assert wait["releasable_bytes_per_stage_rank"] == [[4096.0]]
    assert memory.budget_version == held
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert any(row["kind"] == "grant" and row["name"] == "A"
               and row["event"] is event for row in wait["wake_events"])
    assert memory.release_grant(0, "A", generation=1, event=event) is True
    assert memory.grant_batch(0, [{
        "name": "B", "bytes": 4096, "physical_key": "Q", "storage": "S",
        "ranges": (0, 4096), "generation": 2}]) is True
    assert memory.occupied_bytes(0, 0) == 4096.0


def test_onchip_and_host_tiers_never_invent_hbm_bytes():
    memory = _capacity_memory(0)
    assert memory.capacity == [[0.0]]
    assert memory.grant_batch(0, [{
        "name": "onchip", "bytes": 4096, "tier": "l2"}]) is True
    assert memory.occupied_bytes(0, 0) == 0.0
    assert memory.live_grants(0, 0) == 0.0
    assert memory.host_occupied_bytes(0) == 0.0
    assert memory.grant_bindings[(0, "onchip")]["tier"] == "l2"
    assert memory.release_grant(0, "onchip") is True
    assert memory.occupied_bytes(0, 0) == 0.0

    assert memory.grant_batch(0, [{
        "name": "host", "bytes": 0, "tier": "host",
        "host_bytes": 4096}]) is True
    assert memory.occupied_bytes(0, 0) == 0.0
    assert memory.host_occupied_bytes(0) == 4096.0
    assert memory.release_grant(0, "host") is True
    assert memory.host_occupied_bytes(0) == 0.0

    version = memory.budget_version
    assert memory.grant_batch(0, [{
        "name": "negative", "bytes": 4096, "ranges": (-1, 4096)}]) is False
    assert memory.budget_version == version
    assert memory.grant_bindings == {}
    assert memory.grant_batch(0, [{"name": "hbm", "bytes": 4096}]) is False
    assert memory.occupied_bytes(0, 0) == 0.0


def test_reader_retry_and_same_name_new_generation_keep_each_real_owner():
    memory = _capacity_memory(8192)
    first_event, second_event = object(), object()
    assert memory.grant_batch(0, [{
        "name": "L0", "bytes": 4096, "physical_key": "P", "storage": "S",
        "ranges": (0, 4096), "generation": 3, "event": first_event,
        "rank": 0}]) is True
    assert memory.hold_for_async_consumer(
        0, "L0", "reader", grant="L0", generation=3,
        event=first_event) is True
    version = memory.budget_version
    # The same real handle is idempotent, not a second reader.
    assert memory.hold_for_async_consumer(
        0, "L0", "reader", grant="L0", generation=3,
        event=first_event) is True
    assert memory.budget_version == version
    assert len(memory.grant_readers[(0, "L0")]) == 1
    assert memory.occupied_bytes(0, 0) == 4096.0

    # A conflicting generation under the same consumer name is refused and
    # leaves the first reader's physical reference untouched.
    assert memory.hold_for_async_consumer(
        0, "L0", "reader", grant="L0", generation=4,
        event=second_event) is False
    assert memory.budget_version == version
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_grant(
        0, "L0", generation=3, event=first_event, rank=0) is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_grant(0, "L0", generation=4) is False

    # A new live generation of the same logical name is a distinct physical
    # allocation; the old reader's handle still owns its own range.
    assert memory.grant_batch(0, [{
        "name": "L0", "bytes": 4096, "physical_key": "P", "storage": "S",
        "ranges": (0, 4096), "generation": 4, "event": second_event,
        "rank": 0}]) is True
    assert memory.occupied_bytes(0, 0) == 8192.0
    assert memory.hold_for_async_consumer(
        0, "L0", "reader", grant="L0", generation=4,
        event=second_event) is False
    assert memory.release_async_consumer(
        0, "L0", "reader", generation=3, event=first_event, rank=1) is False
    assert memory.release_async_consumer(
        0, "L0", "reader", generation=3, event=first_event, rank=0) is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.hold_for_async_consumer(
        0, "L0", "reader", grant="L0", generation=4,
        event=second_event) is True
    # The reader aliases the surviving generation-4 grant range.
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_async_consumer(
        0, "L0", "reader", generation=4, event=second_event, rank=0) is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_grant(
        0, "L0", generation=4, event=second_event, rank=1) is False
    assert memory.release_grant(
        0, "L0", generation=4, event=second_event, rank=0) is True
    assert memory.occupied_bytes(0, 0) == 0.0

    cache = _capacity_memory(1024)
    assert cache.resize(0, "r", 4) is True
    assert cache.hold_for_async_consumer(0, "r", "consumer") is True
    held = cache.budget_version
    assert cache.hold_for_async_consumer(0, "r", "consumer") is True
    assert cache.budget_version == held
    assert cache.pending_async_consumers(0, "r") == ("consumer",)
    assert cache.hold_for_async_consumer(
        0, "r", "consumer", generation=3) is False
    assert cache.pending_async_consumers(0, "r") == ("consumer",)
    assert cache.release_async_consumer(0, "r", "consumer", rank=0) is False
    assert cache.release_async_consumer(0, "r", "consumer") is True

    # A reader owns its own last-use Event; it need not be the grant handle's
    # event, and only the reader Event may complete that reference.
    separate = _capacity_memory(4096)
    grant_event, reader_event = object(), object()
    assert separate.grant_batch(0, [{
        "name": "G", "bytes": 4096, "physical_key": "P", "storage": "S",
        "ranges": (0, 4096), "generation": 7,
        "event": grant_event}]) is True
    assert separate.hold_for_async_consumer(
        0, "G", "reader", grant="G", generation=7,
        event=reader_event) is True
    assert separate.release_grant(
        0, "G", generation=7, event=grant_event) is True
    assert separate.occupied_bytes(0, 0) == 4096.0
    assert separate.release_async_consumer(
        0, "G", "reader", generation=7, event=object()) is False
    assert separate.occupied_bytes(0, 0) == 4096.0
    assert separate.release_async_consumer(
        0, "G", "reader", generation=7, event=reader_event) is True
    assert separate.occupied_bytes(0, 0) == 0.0

    # A grant that declared no event cannot be released by an arbitrary one.
    no_event = _capacity_memory(4096)
    assert no_event.grant_batch(0, [{
        "name": "N", "bytes": 4096, "physical_key": "Q",
        "ranges": (0, 4096)}]) is True
    assert no_event.release_grant(0, "N", event=object()) is False
    assert no_event.occupied_bytes(0, 0) == 4096.0
    assert no_event.release_grant(0, "N") is True
    assert no_event.occupied_bytes(0, 0) == 0.0


def test_shared_page_retain_charges_one_physical_page_per_reference():
    memory = _capacity_memory(4096)
    assert memory.retain_shared_pages(
        0, "reader", (4096.0,), physical_key="P", storage="S") is True
    assert memory.retain_shared_pages(
        0, "reader", (4096.0,), physical_key="P", storage="S") is True
    assert memory.shared_pages[(0, "P")]["refs"] == 2
    assert memory.occupied_bytes(0, 0) == 4096.0
    # A second real owner on the same page is another reference, not bytes.
    assert memory.retain_shared_pages(
        0, "reader2", (4096.0,), physical_key="P", storage="S") is True
    assert memory.shared_pages[(0, "P")]["refs"] == 3
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_shared_pages(0, "reader") is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_shared_pages(0, "reader") is True
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert memory.release_shared_pages(0, "reader2") is True
    assert memory.occupied_bytes(0, 0) == 0.0

    # A selected allocation and a grant can alias the same explicit physical
    # page; the union is charged once and remains owned by the survivor.
    cross = _capacity_memory(4096)
    assert cross.attach_allocation_plan(AllocationPlan((
        Allocation("selected:P", 4096, "workspace",
                   physical_key="P", storage="S"),)))
    assert cross.grant_batch(0, [{
        "name": "grant:P", "bytes": 4096, "physical_key": "P",
        "storage": "S", "ranges": (0, 4096)}]) is True
    assert cross.occupied_bytes(0, 0) == 4096.0
    assert cross.peak_combined == [[4096.0]]
    assert cross.selected_reserved == [[4096.0]]
    assert cross.granted == [[0.0]]
    assert cross.release_grant(0, "grant:P") is True
    assert cross.occupied_bytes(0, 0) == 4096.0
    assert cross.release_selected_allocation("selected:P") is True
    assert cross.occupied_bytes(0, 0) == 0.0
