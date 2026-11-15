"""Bounded, identity-preserving MCP wrappers for the opflow engine.

[Role]
Expose catalog, hardware, evaluation, transfer, network, and Triton lowering as
deterministic MCP-style tools with compact schemas that keep `op`, `sku`, and
`mode` intact and stay <= 250 UTF-8 bytes / 250 BPE tokens.
[Invariants]
Every success payload carries `ok`, full machine-readable identity, numeric
results, a stable `ref`, and warning codes. Required identity keys are never
truncated; an over-budget identity is an explicit EBUDGET failure. Unknown
tools/kwargs and fail-closed validation errors return compact codes.
[Timeouts & Locks]
Pure in-memory calls; no locks, waits, I/O, or subprocesses.
[Side-Effect Policy]
Read-only; no files or device resources are touched.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from .flow_schema import AuthoringBackend, BufferPool
from .hardware_profiles import ProfileError, get_profile
from .network_groups import list_network_operator_group
from .paradigm_catalog import (
    UncalibratedOperatorError, build_flow, catalog_info, evaluate_operator,
    list_operators,
)
from .pipeline_evaluator import ExecutionModeError, FlowEvaluation, TilingError
from .token_budget import bounded_envelope, bounded_error
from .transfer_accounting import TransferError, TransferRequest, account_transfer

_WARN_CODES = {
    "stateless_retile_zero_spill": "RETILE",
    "algorithm_workspace_2pass": "PASS2",
    "working_set_spills_l2": "L2",
    "simt_ub_budget_enforced": "SIMT",
    "fixpipe_l0c_to_ub_lane_selected": "FIXUB",
    "1c2v_without_explicit_sync": "NOSYNC",
    "single_buffer_serializes_stage_chain": "SINGLE",
    "structural_schema_only_uncalibrated_prior": "STRUCT",
    "regbase_dual_issue_overlap": "REG",
    "regbase_dual_issue_unavailable_on_mem_3201": "NOREG",
    "zero_launch_noop": "NOOP",
}


def _ok(payload: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    payload.setdefault("ok", 1)
    return bounded_envelope(payload, required=required)


def _err(code: str, message: str = "") -> dict[str, Any]:
    return bounded_error(code, message)


def _ref(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _warn_codes(warnings: Any) -> list[str]:
    codes: list[str] = []
    for warning in warnings:
        text = str(warning)
        code = _WARN_CODES.get(text)
        if code is None:
            code = "W" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:4].upper()
        if code not in codes:
            codes.append(code)
    return codes


def opflow_catalog() -> dict[str, Any]:
    info = catalog_info()
    return _ok(
        {"count": len(info), "groups": len({entry["group"] for entry in info}),
         "frontier": sum(1 for entry in info if entry["frontier"])},
        ("ok", "count", "groups", "frontier"),
    )


def opflow_hardware(name: str = "Ascend950PR", sku: str | None = None) -> dict[str, Any]:
    try:
        profile = get_profile(name, sku=sku)
        profile.require_prediction_ready()
    except ProfileError as exc:
        return _err("EPROFILE", str(exc))
    return _ok(
        {
            "sku": profile.sku, "arch": profile.arch, "aic": profile.aic_cores,
            "aiv": profile.aiv_cores, "freq": profile.freq_mhz,
            "ub": profile.memory.ub_per_lane,
            "simt": profile.ub_usable_bytes("simt") if profile.arch_code == 3510 else 0,
            "ssbuf": profile.ssbuf_bytes(True), "l0c": profile.memory.l0c,
            "cube": round(profile.cube_peak_tflops("bf16"), 1),
            "vec": round(profile.vector_peak_tflops("bf16"), 1),
            "hbm": round(profile.hbm_bw_gbps, 1), "l2": profile.memory.l2,
            "ubfix": int(profile.supports_fixpipe_l0c_to_ub()),
            "est": "derived_peaks",
        },
        ("ok", "sku", "arch", "aic", "aiv", "ub", "simt", "ssbuf", "l0c",
         "cube", "vec", "hbm", "ubfix"),
    )


def _evaluation_payload(result: FlowEvaluation, profile) -> dict[str, Any]:
    return {
        "op": result.name, "sku": profile.sku, "mode": result.mode.value,
        "us": round(result.t_est_us, 4), "tau": round(result.tau_steady_us, 4),
        "est": result.estimate_kind, "cal": int(result.calibrated),
        "warn": _warn_codes(result.warnings),
        "warn_text": [str(warning) for warning in result.warnings],
        "ref": _ref(result.name, profile.sku, result.mode.value, result.t_est_us),
    }


def opflow_evaluate(op: str, name: str = "Ascend950PR", sku: str | None = None,
                    mode: str = "aclgraph_full", *, allow_structural_prior: bool = False,
                    **params: Any) -> dict[str, Any]:
    """Evaluate a catalog operator with strict parameters and full identity."""
    if not isinstance(allow_structural_prior, bool):
        return _err("EARGS", "allow_structural_prior must be bool")
    try:
        profile = get_profile(name, sku=sku)
        result = evaluate_operator(str(op), profile, mode,
                                   allow_structural_prior=allow_structural_prior,
                                   **params)
    except TimeoutError:
        raise
    except UncalibratedOperatorError as exc:
        return _err("EUNCAL", str(exc))
    except (KeyError, ValueError, TilingError, ExecutionModeError) as exc:
        return _err("EEVAL", str(exc))
    except Exception as exc:  # defensive MCP boundary
        return _err("EINTERNAL", type(exc).__name__)
    return _ok(
        _evaluation_payload(result, profile),
        ("ok", "op", "sku", "mode", "us", "tau", "est", "cal", "warn", "ref"),
    )


def opflow_network_group(network: str = "k3") -> dict[str, Any]:
    try:
        names = list_network_operator_group(network)
    except KeyError as exc:
        return _err("EGROUP", str(exc))
    return _ok(
        {"group": str(network).lower(), "count": len(names), "head": names[0], "tail": names[-1]},
        ("ok", "group", "count", "head", "tail"),
    )


def opflow_triton(op: str = "flash_attn_score", name: str = "Ascend950PR",
                  sku: str | None = None, *, allow_structural_prior: bool = False,
                  **features: Any) -> dict[str, Any]:
    if not isinstance(allow_structural_prior, bool):
        return _err("EARGS", "allow_structural_prior must be bool")
    try:
        profile = get_profile(name, sku=sku)
        flow = build_flow(str(op), allow_structural_prior=allow_structural_prior)
        flow = replace(flow, authoring_backend=AuthoringBackend.TRITON_ASCEND)
        from .pipeline_evaluator import evaluate_flow

        result = evaluate_flow(flow, profile, "aclgraph_full", triton_features=features,
                               allow_structural_prior=allow_structural_prior)
    except (KeyError, ValueError, ProfileError) as exc:
        return _err("ETRITON", str(exc))
    detail = result.extra.get("triton", {})
    mfu = result.extra.get("mfu", {})
    return _ok(
        {
            "op": result.name, "sku": profile.sku, "us": round(result.t_est_us, 4),
            "est": result.estimate_kind, "cal": int(result.calibrated),
            "serial": round(float(detail.get("serialism", 1.0)), 3),
            "overlap": round(float(detail.get("observed_overlap", 1.0)), 3),
            "gain": round(float(detail.get("potential_gain", 1.0)), 3),
            "gather": detail.get("gather", "none"),
            "mfu": None if mfu.get("mfu") is None else round(float(mfu["mfu"]), 4),
            "valid": int(mfu.get("valid", False)),
            "warn": _warn_codes(result.warnings),
            "warn_text": [str(warning) for warning in result.warnings],
            "ref": _ref(result.name, profile.sku, "triton", result.t_est_us),
        },
        ("ok", "op", "sku", "us", "est", "cal", "serial", "gather", "mfu", "valid", "ref"),
    )


def opflow_transfer(kind: str, name: str = "Ascend950PR", sku: str | None = None,
                    rows: int = 1, cols: int = 1, element_bytes: int = 2,
                    src_pool: str = "GM", dst_pool: str = "UB_LANE0",
                    read_policy: str = "full_source_once", slots: int = 1) -> dict[str, Any]:
    try:
        profile = get_profile(name, sku=sku)
        request = TransferRequest(
            kind=kind, rows=rows, cols=cols, element_bytes=element_bytes,
            src_pool=BufferPool(src_pool), dst_pool=BufferPool(dst_pool),
            read_policy=read_policy, slots=slots,
        )
        account = account_transfer(request, profile)
    except TimeoutError:
        raise
    except (TransferError, ProfileError, ValueError) as exc:
        return _err("ETRANSFER", str(exc))
    except Exception as exc:
        return _err("EINTERNAL", type(exc).__name__)
    return _ok(
        {
            "kind": account.kind.value, "log": account.logical_bytes,
            "rd": account.actual_read_bytes, "wr": account.actual_write_bytes,
            "cap": account.physical_capacity_bytes, "span": account.address_span_bytes,
            "ref": _ref(account.kind.value, account.logical_bytes, account.actual_read_bytes),
        },
        ("ok", "kind", "log", "rd", "wr", "cap", "span", "ref"),
    )


TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "opflow_catalog", "description": "List opflow operator count/frontier count", "input": {}},
    {"name": "opflow_hardware", "description": "Get canonical Ascend geometry", "input": {"name": "str", "sku": "str?"}},
    {"name": "opflow_evaluate", "description": "Evaluate one op flow and launch floor", "input": {"op": "str", "name": "str", "sku": "str?", "mode": "str"}},
    {"name": "opflow_transfer", "description": "Account logical/actual/physical bytes", "input": {"kind": "str", "rows": "int", "cols": "int"}},
    {"name": "opflow_network_group", "description": "List K3/DS41 network membership", "input": {"network": "str"}},
    {"name": "opflow_triton", "description": "Evaluate Triton lowering/MFU hazards", "input": {"op": "str", "name": "str", "sku": "str?"}},
)

_TOOLS = {
    "opflow_catalog": opflow_catalog, "opflow_hardware": opflow_hardware,
    "opflow_evaluate": opflow_evaluate, "opflow_network_group": opflow_network_group,
    "opflow_triton": opflow_triton, "opflow_transfer": opflow_transfer,
}


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    function = _TOOLS.get(str(name))
    if function is None:
        return _err("ETOOL", str(name))
    if arguments is not None and not isinstance(arguments, dict):
        return _err("EARGS", "arguments must be an object")
    try:
        return function(**(arguments or {}))
    except TimeoutError:
        raise
    except TypeError as exc:
        return _err("EARGS", str(exc))
    except Exception as exc:  # defensive MCP boundary
        return _err("EINTERNAL", type(exc).__name__)


__all__ = [
    "TOOL_SPECS", "call_tool", "opflow_catalog", "opflow_evaluate",
    "opflow_hardware", "opflow_network_group", "opflow_transfer", "opflow_triton",
]
