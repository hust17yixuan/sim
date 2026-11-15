"""Built-in CANNBot paradigm, frontier, and K3/DS41 operator registry.

[Role]
Register every source-backed builder and every catalog-only identifier, tag
network membership, and fail closed with a typed UncalibratedOperatorError when
a caller asks for microsecond predictions from metadata-only catalog entries.
[Invariants]
Names are unique and sorted on output. A catalog entry never implies semantic
equivalence or a calibrated formula. Prediction endpoints require `allow_structural_prior=True` for metadata-only
or conditional routes. Raw schemas remain available for labeled analysis.
[Timeouts & Locks]
Arithmetic-only; no device calls, sleeps, locks, or external dependencies.
[Side-Effect Policy]
Read-only module; builders allocate in-memory records only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from .catalog_attention import (
    flash_attn, flash_attn_fp8_fullquant, flash_attn_score, flash_kda,
    flash_kda_fwd, fused_recurrent_kda_snapshot, mega_full_index, mega_mla,
    mega_recurrent_kda, moonep, sparse_flash_mla,
)
from .catalog_common import check_kwargs, make_flow, pick_int, ub_buffers
from .catalog_flows import (
    add, add_rms_norm, attn_epilogue, block_attn_res_update_rms_norm,
    commit_recurrent_kda, dynamic_mx_quant_sr, engram_gate,
    flash_kda_cp_preprocess, fused_recurrent_kda_replayssm, gather_v2,
    gdn2_prepare, index_gather, layernorm_welford, mhc_hcpost,
    mhc_hcpre_cv_splitk, rejection_random_sampling, rms_norm, scatter_elements,
    softmax_long_r, swiglu_dynamic_quant, top_k_mrgsort,
)
from .catalog_matmul import (
    batch_matmul, dual_level_quant_matmul, fused_matmul, grouped_matmul,
    grouped_matmul_finalize_routing, matmul, matmul_bf16, matmul_streamk,
    quant_batch_matmul_inplace_add, quant_batch_matmul_mxa8w4,
    quant_batch_matmul_w4a8_mxfp, quant_matmul, transpose_batch_matmul,
    weight_quant_batchmatmul,
)
from .flow_schema import BufferPool, ExecutionArchetype, OpFlowTemplate, PipeWork
from .hardware_profiles import HardwareProfile
from .network_groups import (
    DS41_OPERATORS, K3_OPERATORS, build_network_flow, network_membership,
)
from .v4_math import classification
from .pipeline_evaluator import FlowEvaluation, evaluate_flow

Builder = Callable[..., OpFlowTemplate]
_OMITTED = object()


class UncalibratedOperatorError(ValueError):
    """A catalog-only entry has no source-faithful flow formula yet."""


STRUCTURAL_ONLY: frozenset[str] = frozenset({
    "fused_norm_rope_attn_rope_cast_fwd", "attn_prologue", "indexer_prologue_k",
    "indexer_prologue_qw", "quant_block_sparse_attn", "qwen_sparse_attn",
    "stem_indexer", "qsa_indexer", "quant_lightning_indexer",
    "quant_lightning_indexer_metadata", "quant_sparse_lightning_indexer",
    "quant_sparse_lightning_indexer_metadata", "mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata", "sparse_flash_mla_metadata",
    "quant_mla_prolog", "mla_prolog_v4", "mega_reindex", "mega_compressor",
    "flash_kda_cp_main", "flash_mla_with_kvcache",
})

_BUILDERS: dict[str, Builder] = {
    "add": add, "add_rms_norm": add_rms_norm, "attn_epilogue": attn_epilogue,
    "batch_matmul": batch_matmul,
    "block_attn_res_update_rms_norm": block_attn_res_update_rms_norm,
    "commit_recurrent_kda": commit_recurrent_kda,
    "dual_level_quant_matmul": dual_level_quant_matmul,
    "dynamic_mx_quant_sr": dynamic_mx_quant_sr, "engram_gate": engram_gate,
    "flash_attn": flash_attn, "flash_attn_fp8_fullquant": flash_attn_fp8_fullquant,
    "flash_attn_score": flash_attn_score, "flash_kda": flash_kda,
    "flash_kda_cp_preprocess": flash_kda_cp_preprocess,
    "flash_kda_fwd": flash_kda_fwd, "fused_matmul": fused_matmul,
    "fused_recurrent_kda_replayssm": fused_recurrent_kda_replayssm,
    "fused_recurrent_kda_snapshot": fused_recurrent_kda_snapshot,
    "gather_v2": gather_v2, "gdn2_prepare": gdn2_prepare,
    "grouped_matmul": grouped_matmul,
    "grouped_matmul_finalize_routing": grouped_matmul_finalize_routing,
    "index_gather": index_gather, "layernorm_welford": layernorm_welford,
    "matmul": matmul, "matmul_bf16": matmul_bf16, "matmul_streamk": matmul_streamk,
    "mega_full_index": mega_full_index, "mega_mla": mega_mla,
    "mega_recurrent_kda": mega_recurrent_kda, "mhc_hcpost": mhc_hcpost,
    "mhc_hcpre_cv_splitk": mhc_hcpre_cv_splitk, "moonep": moonep,
    "quant_batch_matmul_inplace_add": quant_batch_matmul_inplace_add,
    "quant_batch_matmul_mxa8w4": quant_batch_matmul_mxa8w4,
    "quant_batch_matmul_w4a8_mxfp": quant_batch_matmul_w4a8_mxfp,
    "quant_matmul": quant_matmul, "rejection_random_sampling": rejection_random_sampling,
    "rms_norm": rms_norm, "scatter_elements": scatter_elements,
    "softmax_long_r": softmax_long_r, "sparse_flash_mla": sparse_flash_mla,
    "swiglu_dynamic_quant": swiglu_dynamic_quant, "top_k_mrgsort": top_k_mrgsort,
    "transpose_batch_matmul": transpose_batch_matmul,
    "weight_quant_batchmatmul": weight_quant_batchmatmul,
}

_VECTOR_NORM = {
    "add", "add_rms_norm", "attn_epilogue", "block_attn_res_update_rms_norm",
    "dynamic_mx_quant_sr", "engram_gate", "gdn2_prepare", "layernorm_welford",
    "mhc_hcpost", "rms_norm", "swiglu_dynamic_quant",
}
_REDUCTION = {
    "gather_v2", "index_gather", "rejection_random_sampling", "scatter_elements",
    "softmax_long_r", "top_k_mrgsort",
}
_CUBE = {
    "batch_matmul", "dual_level_quant_matmul", "fused_matmul", "grouped_matmul",
    "grouped_matmul_finalize_routing", "matmul", "matmul_bf16", "matmul_streamk",
    "quant_batch_matmul_inplace_add", "quant_batch_matmul_mxa8w4",
    "quant_batch_matmul_w4a8_mxfp", "quant_matmul", "transpose_batch_matmul",
    "weight_quant_batchmatmul",
}
_FRONTIER = {
    "block_attn_res_update_rms_norm", "commit_recurrent_kda", "engram_gate",
    "flash_attn", "flash_attn_fp8_fullquant", "flash_attn_score", "flash_kda",
    "flash_kda_fwd", "fused_recurrent_kda_replayssm",
    "fused_recurrent_kda_snapshot", "grouped_matmul_finalize_routing",
    "matmul_streamk", "mega_full_index", "mega_mla", "mega_recurrent_kda",
    "mhc_hcpost", "mhc_hcpre_cv_splitk", "moonep", "quant_batch_matmul_mxa8w4",
    "quant_batch_matmul_w4a8_mxfp", "sparse_flash_mla",
}


def _structural_prior(name: str, **kw: Any) -> OpFlowTemplate:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "m", "n", "k", "seq"})
    tokens = pick_int(kw, ("tokens", "rows"), 32)
    hidden = pick_int(kw, ("hidden", "cols"), 2048)
    elements = tokens * hidden
    read = elements * 4
    write = elements * 2
    return make_flow(
        name, ExecutionArchetype.PURE_VECTOR, elements, 4096,
        (PipeWork("VEC", float(elements) * 12.0),),
        gm_read_bytes=read, gm_write_bytes=write, unique_gm_bytes=read + write,
        tiling_pool=BufferPool.UB_LANE0,
        live_bytes_per_unit=6.0, buffers=ub_buffers(4096, 2, lanes=2),
        params={"tokens": tokens, "hidden": hidden, "prior": True},
        notes="structural_schema_only: catalog metadata has no encoded formula; uncalibrated",
        estimate_kind="structural_schema_only", calibrated=False,
    )


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    group: str
    builder: Builder
    paradigm: str
    frontier: bool = False
    networks: tuple[str, ...] = ("common",)

    def __post_init__(self) -> None:
        if not isinstance(self.frontier, bool):
            raise ValueError("frontier must be bool")


def _group(name: str) -> str:
    if name in _VECTOR_NORM:
        return "vector_norm"
    if name in _REDUCTION:
        return "reduction_index"
    if name in _CUBE:
        return "cube_quant"
    return "cv_attention"


def _entry(name: str, builder: Builder) -> CatalogEntry:
    return CatalogEntry(_group(name), builder, _group(name), name in _FRONTIER,
                        network_membership(name))


CATALOG: dict[str, CatalogEntry] = {}
for _name, _builder in sorted(_BUILDERS.items()):
    CATALOG[_name] = _entry(_name, _builder)
for _name in sorted(STRUCTURAL_ONLY):
    CATALOG[_name] = _entry(_name, _structural_prior)
_MISSING_NETWORK = (set(K3_OPERATORS) | set(DS41_OPERATORS)) - set(CATALOG)
if _MISSING_NETWORK:
    raise RuntimeError(f"network operators missing builders: {sorted(_MISSING_NETWORK)}")


def list_operators(group: str | None = None, network: str | None = None) -> tuple[str, ...]:
    return tuple(
        name for name, entry in sorted(CATALOG.items())
        if (group is None or entry.group == group)
        and (network is None or network in entry.networks)
    )


def catalog_info() -> list[dict[str, Any]]:
    return [
        {
            "name": name, "group": entry.group, "paradigm": entry.paradigm,
            "frontier": entry.frontier, "networks": list(entry.networks),
            "estimate_kind": ("structural_schema_only" if name in STRUCTURAL_ONLY
                              else "analytic_prior"),
            **classification(name),
        }
        for name, entry in sorted(CATALOG.items())
    ]


def build_flow(name: str, *, allow_structural_prior: bool | object = _OMITTED,
               **params: Any) -> OpFlowTemplate:
    allow = allow_structural_prior
    if allow is not _OMITTED and not isinstance(allow, bool):
        raise ValueError("allow_structural_prior must be bool")
    try:
        entry = CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"unknown operator flow: {name!r}") from exc
    if name in STRUCTURAL_ONLY:
        if allow is False:
            raise UncalibratedOperatorError(
                f"{name!r} is catalog metadata only; no source-exact flow formula "
                "is encoded. Pass allow_structural_prior=True to receive an "
                "explicitly uncalibrated structural prior."
            )
        # An omitted permission may construct the clearly labelled
        # uncalibrated schema for analysis; the prediction API passes False and
        # fails closed. The result always carries calibrated=False + a warning.
        flow = _structural_prior(name, **params)
        return replace(flow, params=dict(flow.params, **classification(name)))
    flow = entry.builder(**params)
    # Builder provenance belongs to the underlying route, including aliases.
    tags = dict(classification(name), **flow.params)
    kind = ("conditional_ledger_required" if tags["derivation_coverage"] ==
            "conditional_ledger_required" else flow.estimate_kind)
    if kind == "conditional_ledger_required" and allow is False:
        raise UncalibratedOperatorError(f"{name}: conditional ledger required; request an explicitly tagged prior")
    return replace(flow, estimate_kind=kind, calibrated=False, params=tags)


_EVAL_KEYS = frozenset({
    "requested_tile_units", "double_buffer", "active_units", "static_decode",
    "triton_features", "capturable_ops", "dynamic_tiling_host_sync",
})


def evaluate_operator(name: str, profile: HardwareProfile,
                      execution_mode: str = "aclgraph_full", **params: Any) -> FlowEvaluation:
    allow = params.pop("allow_structural_prior", False)
    if not isinstance(allow, bool):
        raise ValueError("allow_structural_prior must be bool")
    build_params = {key: value for key, value in params.items() if key not in _EVAL_KEYS}
    flow = build_flow(name, allow_structural_prior=allow, **build_params)
    forwarded = {key: value for key, value in params.items() if key in _EVAL_KEYS}
    return evaluate_flow(flow, profile, execution_mode, allow_structural_prior=allow,
                         **forwarded)


__all__ = [
    "CATALOG", "CatalogEntry", "DS41_OPERATORS", "K3_OPERATORS",
    "STRUCTURAL_ONLY", "UncalibratedOperatorError", "build_flow",
    "build_network_flow", "catalog_info", "evaluate_operator",
    "list_operators",
]
