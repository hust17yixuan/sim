"""Deterministic Triton-Ascend 33-point optimization and hazard catalog.

[Role]
Encode the ordered `triton-latency-optimizer` 33-point catalog and the
`triton-mfu-analyzer` five-gate validity contract as immutable records consumed
by the lowering evaluator and the knowledge distiller.
[Invariants]
Exactly 33 numbered rules (1..33), unique slugs, stable order. Rule 4 owns the
three random-gather-to-dot paths, 28 owns CV ping-pong/serialism, 30 owns
BLOCK_QO non-affine-store hazards, and 31/32/33 are terminal rules.
[Timeouts & Locks]
Pure module constants and linear lookups; no I/O, locks, or waits.
[Side-Effect Policy]
Read-only constants; no filesystem or device effect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TritonRule:
    number: int
    slug: str
    title: str
    category: str
    hazard_tags: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.number >= 31


_RULE_ROWS = (
    ("constexpr_parameters", "入参静态化", "compile", ()),
    ("tiling_optimization", "Tiling 优化", "memory", ("strided_access",)),
    ("vector_core_partition", "分核优化", "parallel", ("grid_core_mismatch",)),
    ("discrete_memory_access", "离散访存优化", "memory", ("random_gather_to_dot",)),
    ("scalar_to_vector", "Scalar 转 Vector", "vector", ("scalar_fallback",)),
    ("avoid_scalar_lowering", "避免向量 API 标量降级", "vector", ("i64_mask", "noncontig_reduce")),
    ("pass_merge", "Pass 消除合并", "memory", ("redundant_pass",)),
    ("dimension_merge", "维度合并", "loop", ("nested_loop",)),
    ("libdevice_usage", "Libdevice 函数使用", "math", ("hand_math",)),
    ("loop_invariant_hoisting", "循环不变量外提", "loop", ("redundant_load",)),
    ("load_order", "Load 指令重排序", "memory", ("load_dependency_bubble",)),
    ("grid_dispatch_specialization", "Grid 形状与多路径特化", "dispatch", ("one_kernel_many_shapes",)),
    ("mixed_strategy", "混合策略自动选择", "dispatch", ("shape_dtype_branch",)),
    ("dimension_merge_large_block", "维度合并与大 BLOCK 累加", "reduction", ("low_mask_coverage",)),
    ("continuous_copy_aggregation", "连续拷贝聚合", "copy", ("fine_grained_copy",)),
    ("redundant_boundary_operation", "消除冗余边界运算", "vector", ("redundant_mask",)),
    ("kernel_splitting", "Kernel 分裂", "dispatch", ("multi_case_slow",)),
    ("cube_mte3_decoupling", "Cube/MTE3 分阶段解耦", "cv", ("atomic_reduction",)),
    ("host_tensor_concat", "Host 侧张量维度拼接", "host", ("split_dot",)),
    ("workspace_decoupling", "Workspace 物化解耦", "memory", ("repeat_gather",)),
    ("latency_bound_tile_merge", "Latency-Bound Tile 合并", "loop", ("small_dot",)),
    ("device_side_gather", "Device-side Gather 连续化", "memory", ("workspace_roundtrip",)),
    ("chained_matmul_buffer_dtype", "Matmul 链中间 buffer dtype", "cv", ("fp32_acc_spill",)),
    ("preinitialized_output_optimization", "输出预初始化", "vector", ("default_fill_branch",)),
    ("ascend_interpolate_optimization", "Interpolate 专用优化", "operator", ("runtime_coord",)),
    ("ascend_pooling_optimization", "Pooling 专用优化", "operator", ("flat_index",)),
    ("matmul_transpose", "Matmul Transpose 专用优化", "operator", ("transpose_strided",)),
    ("cv_fusion", "CV 融合优化", "cv", ("cv_serialism", "gm_intermediate_fallback")),
    ("flash_attention_optimization", "Attention/FA 专用优化", "attention", ("serial_ratio_gt_1_5",)),
    ("mla_paged_attention_optimization", "Attention/MLA 专用优化", "attention", ("mla_block_qo",)),
    ("ir_analysis_workflow", "IR 分析", "terminal", ("missing_ir_evidence",)),
    ("autotune", "Autotune 自动调优", "terminal", ("untuned_constexpr",)),
    ("block_size_scaling", "Block Size Scaling", "terminal", ("unscaled_block",)),
)

TRITON_RULES: tuple[TritonRule, ...] = tuple(
    TritonRule(index + 1, slug, title, category, tags)
    for index, (slug, title, category, tags) in enumerate(_RULE_ROWS)
)

MFU_GATES: tuple[str, ...] = (
    "input_budget_ok",
    "model_ref_valid",
    "implementation_runs",
    "single_run_time_budget_ok",
    "effective_bw_within_hbm",
)

GATHER_PATHS: tuple[str, ...] = (
    "none",
    "gather_out_to_ub",
    "in_kernel_l2_stage",
    "device_gather_workspace",
)


def rule_by_number(number: int) -> TritonRule:
    if not 1 <= int(number) <= len(TRITON_RULES):
        raise KeyError(f"unknown Triton rule number: {number!r}")
    return TRITON_RULES[int(number) - 1]


def rules_by_tag(tag: str) -> tuple[TritonRule, ...]:
    return tuple(rule for rule in TRITON_RULES if tag in rule.hazard_tags)


__all__ = [
    "GATHER_PATHS",
    "MFU_GATES",
    "TRITON_RULES",
    "TritonRule",
    "rule_by_number",
    "rules_by_tag",
]
