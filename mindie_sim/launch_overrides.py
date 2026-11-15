"""Version-pinned action handlers for launch declarations.

[Role]
  Resolve one literal CLI/env occurrence into Case patches, required inputs or
  an explicit unmodeled rule.  `launch_bindings` owns the report lifecycle;
  this module owns the version-sensitive action semantics.

[Invariants]
  * MLAPO enable/disable is interpreted only through the source-specific
    adapter spelling: the 0.30 config switch never authorizes the 0.18 env
    spelling and vice versa.
  * `--enforce-eager` actively clears ACLGraph capture state (mode, keys,
    piecewise keys and captured execution metadata).
  * JSON parents keep unique occurrence IDs; unknown children are unmodeled
    while known children still bind.  Duplicate JSON keys are structured
    failures, never uncaught type errors.
  * `peak_ms` is service only for a declared resource rate; the rate is never
    assumed to be one.
  * Secret provenance is applied by every consumer, including the override
    ledger and diagnostic text.

[Timeouts & Locks]
  Bounded in-memory table lookup and string scans; no locks.

[Side-Effect Policy]
  No I/O, environment lookup, subprocess or eval.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .compose import _resolve_va_lowerings
from .launch_actions import (
    BindingFailure, Draft, Occurrence, UnsupportedModeling, _ALL2ALL, _DELETE,
    _BOOL, _JSON_FLAGS, _METHODS, _OPERATIONAL, _SCALAR, _child, _draft, _scrub,
)
from .launch_documents import DocumentInputError, finite_json_loads
from .launch_json_paths import json_member_paths
from .launch_parser import is_secret_name
from .schema.common import ModelingError


def _eager_patches():
    """An eager declaration removes every ACLGraph capture/replay record."""
    return {"/runtime/graph_mode": "none",
            "/runtime/graph_capture_mode": "none",
            "/runtime/graph_capture_keys": [],
            "/runtime/piecewise_capture_keys": [],
            "/runtime/mixed_mode": "none",
            "/runtime/metadata/captured_execution_mode": "NONE"}


def _eager_conflict(drafts):
    """Deterministic precedence: ``enforce_eager=True`` wins over capture.

    vLLM's ``enforce_eager=True`` runs eagerly regardless of the CLI order of
    ``--compilation-config``.  The compilation parent and every child become
    shadowed by the eager occurrence, so no report can show an effective eager
    setting alongside ``graph_mode="full"``.
    """
    eager = next((draft for draft in drafts
                  if draft.key == "enforce_eager" and draft.effective
                  and draft.bucket == "applied"), None)
    if eager is None:
        return tuple(drafts), ()
    graph = next((draft for draft in drafts
                  if draft.kind == "cli" and draft.name == "--compilation-config"
                  and draft.effective), None)
    if graph is None:
        return tuple(drafts), ()
    requests_capture = any(
        path == "/runtime/graph_mode" and value in {"full", "piecewise"}
        for path, value in graph.patches)
    if not requests_capture:
        return tuple(drafts), ()
    result = []
    for draft in drafts:
        if draft is graph or draft.parent_uid == graph.uid:
            result.append(replace(
                draft, effect="shadowed", effective=False,
                shadowed_by=eager.uid, patches=()))
        else:
            result.append(draft)
    return tuple(result), (graph.uid,)


def _mlapo(context, source):
    if context["adapter"] is None:
        raise UnsupportedModeling("framework_version_unresolved_mlapo",
                                  "MLAPO spelling is version-sensitive; pin a supported release.")
    if source == "env" and not context["adapter"].mlaapo_env:
        raise UnsupportedModeling("legacy_mlapo_env_not_interpreted",
                                  "The pinned adapter no longer interprets the legacy env switch.")
    if source == "config" and not context["adapter"].mlaapo_config:
        raise UnsupportedModeling("enable_mlapo_config_not_in_adapter",
                                  "The pinned adapter does not select MLAPO from additional_config.")
    parsed, base = context.get("parsed"), context.get("base")
    if parsed is None or base is None:
        raise UnsupportedModeling("base_configuration_required",
                                  "Model/hardware configuration is required for the MLAPO domain.")
    try:
        table = _resolve_va_lowerings(parsed.model, base["runtime"]["parallel"],
                                      parsed.machine.sku, parsed.runtime.lowering_passes)
    except (ModelingError, ValueError):
        table = {}
    if "mla_prolog" not in table:
        raise UnsupportedModeling("mlapo_domain_unsupported",
                                  "No mla_prolog lowering in this domain.")
    return (("/runtime/kernels/lowerings/mla_prolog", table["mla_prolog"]),)


def _disable_mlapo(context, source="config"):
    """Disable MLAPO only through a spelling the pinned adapter declares."""
    if context.get("parsed") is None:
        raise UnsupportedModeling("base_configuration_required",
                                  "Model/hardware configuration is required to disable MLAPO.")
    adapter = context["adapter"]
    if adapter is None:
        raise UnsupportedModeling("framework_version_unresolved_mlapo",
                                  "MLAPO spelling is version-sensitive; pin a supported release.")
    if source == "env" and not adapter.mlaapo_env:
        raise UnsupportedModeling("legacy_mlapo_env_not_interpreted",
                                  "The pinned adapter no longer interprets the legacy env switch.")
    if source == "config" and not adapter.mlaapo_config:
        raise UnsupportedModeling("mlapo_disable_not_in_adapter",
                                  "The pinned adapter has no additional_config MLAPO switch.")
    return (("/runtime/kernels/lowerings/mla_prolog", _DELETE),
            ("/runtime/metadata/enable_mlapo", False))


def _flashcomm(context, enable=True):
    if context.get("parsed") is None:
        raise UnsupportedModeling("base_configuration_required",
                                  "Model/hardware configuration is required for FLASHCOMM1.")
    name = context["parsed"].model.name.lower()
    if "qwen3.5" not in name and "qwen3_5" not in name:
        raise UnsupportedModeling("flashcomm1_model_domain",
                                  "Source predicate admits only qwen3_5_text.")
    axes = context["axes"]
    if axes["tp"] <= 1 or axes["pp"] != 1:
        raise UnsupportedModeling("flashcomm1_parallel_predicate",
                                  "Source predicate requires TP>1 and PP=1.")
    if not enable:
        return (("/runtime/kernels/sequence_parallel", _DELETE),)
    try:
        from .prediction.sequence_parallel import DECLARATION
    except (ImportError, ValueError) as exc:
        raise UnsupportedModeling("flashcomm1_declaration_unavailable",
                                  "Declaration unavailable.") from exc
    return (("/runtime/kernels/sequence_parallel",
             dict(DECLARATION, padding="right_to_tp")),)


def _all2all(item, context):
    parsed = context.get("parsed")
    if parsed is None:
        raise UnsupportedModeling("base_configuration_required",
                                  "Model configuration is required for all2all selection.")
    if parsed.model.num_experts <= 1:
        raise UnsupportedModeling("all2all_non_moe",
                                  "No routed-expert branch exists in the selected model.")
    value = str(item.raw).lower()
    if value not in _ALL2ALL:
        raise UnsupportedModeling("all2all_backend_unknown",
                                  "Only MC2/DeepEP/alltoall recipes have consumers.")
    recipe, high = _ALL2ALL[value]
    patches = [("/runtime/moe_recipe", recipe)]
    if high or value in {"deepep", "deepep_high_throughput"}:
        patches.append(("/runtime/deepep_high_throughput", high))
    return tuple(patches)


def _compilation(payload, context, occurrence):
    known = {"cudagraph_mode", "capture_sizes", "cuda_graph_sizes", "splitting_ops",
             "full_cuda_graph"}
    unknown = set(payload) - known
    if unknown:
        raise BindingFailure("compilation_config_unknown_child")
    adapter = context["adapter"]
    if adapter is None:
        raise UnsupportedModeling("framework_version_unresolved_compilation",
                                  "Compilation defaults are version-sensitive; pin a supported release.")
    mode = payload.get("cudagraph_mode", "NONE")
    if not isinstance(mode, str):
        raise BindingFailure("invalid_cudagraph_mode")
    sizes = payload.get("capture_sizes", payload.get("cuda_graph_sizes"))
    if sizes is not None and (not isinstance(sizes, list)
                              or not all(isinstance(item, int) and not isinstance(item, bool)
                                         and item > 0 for item in sizes)):
        raise BindingFailure("invalid_capture_sizes")
    sizes = sorted(set(sizes)) if sizes else None
    mode = mode.upper()
    if mode in {"NONE", ""}:
        patches = _eager_patches()
    elif mode == "PIECEWISE":
        keys = sizes or list(adapter.capture_defaults)
        patches = {"/runtime/graph_mode": "piecewise",
                   "/runtime/graph_capture_mode": "piecewise",
                   "/runtime/piecewise_capture_keys": keys,
                   "/runtime/graph_capture_keys": [], "/runtime/mixed_mode": "none"}
    elif mode in {"FULL", "FULL_DECODE_ONLY"}:
        keys = sizes or list(adapter.capture_defaults)
        if not keys:
            raise BindingFailure("capture_sizes_required")
        patches = {"/runtime/graph_mode": "full", "/runtime/graph_capture_mode": "full",
                   "/runtime/graph_capture_keys": keys,
                   "/runtime/piecewise_capture_keys": [], "/runtime/mixed_mode": "none",
                   "/runtime/metadata/captured_execution_mode": mode}
    else:
        raise BindingFailure("invalid_cudagraph_mode")
    if "splitting_ops" in payload:
        if (not isinstance(payload["splitting_ops"], list)
                or not all(isinstance(item, str) for item in payload["splitting_ops"])):
            raise BindingFailure("invalid_splitting_ops")
        patches["/runtime/splitting_ops"] = list(payload["splitting_ops"])
    if "full_cuda_graph" in payload:
        if not isinstance(payload["full_cuda_graph"], bool):
            raise BindingFailure("invalid_full_cuda_graph")
        patches["/runtime/invalid_full_modes"] = not payload["full_cuda_graph"]
    mapped = {key: ((path, patches[path]),) for key, path in {
        "cudagraph_mode": "/runtime/graph_capture_mode",
        "capture_sizes": "/runtime/graph_capture_keys",
        "cuda_graph_sizes": "/runtime/graph_capture_keys",
        "splitting_ops": "/runtime/splitting_ops",
        "full_cuda_graph": "/runtime/invalid_full_modes"}.items() if path in patches}
    return tuple(patches.items()), mapped


def _additional(payload, context):
    known = {"enable_mlapo", "enable_dsa_cp", "multistream_overlap_shared_expert",
             "enable_flashcomm1"}
    unknown = set(payload) - known
    if unknown:
        raise BindingFailure("additional_config_unknown_child")
    for key in known & set(payload):
        if not isinstance(payload[key], bool):
            raise BindingFailure(f"invalid_boolean:{key}")
    patches, mapped = [], {}
    if "enable_mlapo" in payload:
        selected = (_mlapo(context, "config") if payload["enable_mlapo"]
                    else _disable_mlapo(context, "config"))
        patches.extend(selected)
        mapped["enable_mlapo"] = selected
    if "enable_flashcomm1" in payload:
        selected = _flashcomm(context, payload["enable_flashcomm1"])
        patches.extend(selected)
        mapped["enable_flashcomm1"] = selected
    if "multistream_overlap_shared_expert" in payload:
        value = payload["multistream_overlap_shared_expert"]
        patches.append(("/runtime/multistream_overlap_shared_expert", value))
        mapped["multistream_overlap_shared_expert"] = (
            ("/runtime/multistream_overlap_shared_expert", value),)
    if "enable_dsa_cp" in payload:
        value = payload["enable_dsa_cp"]
        patches.append(("/runtime/parallel/enable_dsa_cp", value))
        mapped["enable_dsa_cp"] = (("/runtime/parallel/enable_dsa_cp", value),)
    return tuple(patches), mapped


def _speculative(payload, context):
    known = {"method", "num_speculative_tokens"}
    if set(payload) - known:
        raise BindingFailure("speculative_config_unknown_child")
    patches, mapped, required = [], {}, []
    if "method" in payload:
        if not isinstance(payload["method"], str) or payload["method"] not in _METHODS:
            raise BindingFailure("unsupported_speculative_method")
        value = _METHODS[payload["method"]]
        patches.append(("/runtime/metadata/draft_mode", value))
        mapped["method"] = (("/runtime/metadata/draft_mode", value),)
    if "num_speculative_tokens" in payload:
        tokens = payload["num_speculative_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise BindingFailure("invalid_num_speculative_tokens")
        patches.append(("/runtime/metadata/draft_proposal_tokens", tokens))
        mapped["num_speculative_tokens"] = (
            ("/runtime/metadata/draft_proposal_tokens", tokens),)
    if context.get("speculative") is None:
        required.append(_required("speculative_acceptance",
                                  "Acceptance marginals/distribution are required inputs.",
                                  ("tpot",)))
    return tuple(patches), mapped, tuple(required)


def _required(name, reason, metrics=("ttft", "tpot", "throughput")):
    from .feature_reports import RequiredInput
    return RequiredInput(name, reason, tuple(metrics))


def _json(item, context, secrets):
    try:
        payload = finite_json_loads(str(item.raw), item.span.source)
    except DocumentInputError as exc:
        raise BindingFailure(f"invalid_json_config:{exc.code}") from exc
    if not isinstance(payload, dict):
        raise BindingFailure("json_config_requires_object")
    pointer = "/" + item.name.lstrip("-").replace("-", "_")
    known = {
        "--compilation-config": {"cudagraph_mode", "capture_sizes", "cuda_graph_sizes",
                                 "splitting_ops", "full_cuda_graph"},
        "--additional-config": {"enable_mlapo", "enable_dsa_cp",
                                "multistream_overlap_shared_expert", "enable_flashcomm1"},
        "--speculative-config": {"method", "num_speculative_tokens"},
        "--kv-transfer-config": set(payload),
    }[item.name]
    unknown = set(payload) - known
    display = _scrub(payload, secrets)
    paths = json_member_paths(str(item.raw))
    if unknown:
        parent = _draft(item, "unmodeled", "json_unknown_child", pointer=pointer,
                        value=display,
                        reason="Outer flag is unsupported as a whole: " + ",".join(sorted(unknown)),
                        secrets=secrets)
        children = tuple(_child(item, key, value, pointer, None, (), secrets,
                                paths)
                         for key, value in payload.items())
        return (parent, *children), ()
    if item.name == "--kv-transfer-config":
        if "kv_connector" not in payload or "kv_role" not in payload:
            raise BindingFailure("kv_transfer_config_incomplete")
        parent = _draft(item, "unmodeled", "vllm.kv_transfer_peer_required",
                        value=display,
                        reason="Complete connector plus peer topology is required; "
                               "no zero transfer is fabricated.", secrets=secrets)
        return (parent,), (_required("kv_transfer_peer",
                                     "A single server Case cannot stand in for a missing peer.",
                                     ("ttft", "tpot", "throughput")),)
    try:
        if item.name == "--compilation-config":
            patches, mapped = _compilation(payload, context, item)
        elif item.name == "--additional-config":
            patches, mapped = _additional(payload, context)
        else:
            patches, mapped, required = _speculative(payload, context)
            parent = _draft(item, "applied" if patches else "operational",
                            "vllm.json_config" if patches else None, patches=patches,
                            pointer=pointer, value=display,
                            targets=tuple(path for path, _ in patches),
                            reason="Existing selected-Case consumer." if patches else None,
                            secrets=secrets)
            children = tuple(_child(item, key, value, pointer,
                                    (mapped[key][0][0] if mapped.get(key) else None),
                                    mapped.get(key, ()), secrets, paths)
                             for key, value in payload.items())
            return (parent, *children), required
    except UnsupportedModeling as exc:
        parent = _draft(item, "unmodeled", exc.rule, pointer=pointer,
                        value=display, reason=exc.reason, secrets=secrets)
        return (parent, *tuple(_child(item, key, value, pointer, None, (), secrets,
                                     paths)
                               for key, value in payload.items())), ()
    if not patches:
        return (_draft(item, "operational", None, value=display, pointer=pointer,
                       secrets=secrets),), ()
    parent = _draft(item, "applied", "vllm.json_config", patches=patches, pointer=pointer,
                    value=display, targets=tuple(path for path, _ in patches),
                    reason="Existing selected-Case consumer.", secrets=secrets)
    children = tuple(_child(item, key, value, pointer,
                            (mapped[key][0][0] if mapped.get(key) else None),
                            mapped.get(key, ()), secrets, paths)
                     for key, value in payload.items())
    return (parent, *children), ()


def _bool_value(item, default):
    raw = item.raw
    if raw is True or raw is False:
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise BindingFailure(f"invalid_boolean_value:{item.name}")


def _resolve_env(item, context, secrets):
    name = item.name
    if is_secret_name(name):
        return (_draft(item, "operational", None, value="[REDACTED]", redacted=True,
                       secrets=secrets),), ()
    if name in {"VLLM_USE_V1", "ASCEND_RT_VISIBLE_DEVICES"}:
        return (_draft(item, "operational", None, value=_scrub(item.raw, secrets),
                       secrets=secrets),), ()
    if name == "VLLM_ASCEND_ENABLE_MLAPO":
        enabled = _bool_value(item, True)
        patches = _mlapo(context, "env") if enabled else _disable_mlapo(context, "env")
        return (_draft(item, "applied", "mlapo.selected_lowering" if enabled
                       else "mlapo.disabled", patches=patches,
                       targets=tuple(path for path, _ in patches),
                       value=enabled, secrets=secrets),), ()
    if name == "VLLM_ASCEND_ENABLE_FLASHCOMM1":
        enabled = _bool_value(item, True)
        patches = _flashcomm(context, enabled)
        return (_draft(item, "applied", "flashcomm1.sequence_parallel" if enabled
                       else "flashcomm1.disabled", patches=patches,
                       targets=tuple(path for path, _ in patches),
                       value=enabled, secrets=secrets),), ()
    if name == "VLLM_ALL2ALL_BACKEND":
        patches = _all2all(item, context)
        return (_draft(item, "applied", "vllm.moe_recipe", patches=patches,
                       targets=tuple(path for path, _ in patches), secrets=secrets),), ()
    if name == "VLLM_ASCEND_ENABLE_DSA_CP":
        enabled = _bool_value(item, True)
        return (_draft(item, "applied", "vllm.dsa_cp",
                       patches=(("/runtime/parallel/enable_dsa_cp", enabled),),
                       targets=("/runtime/parallel/enable_dsa_cp",), value=enabled,
                       secrets=secrets),), ()
    if name.startswith(("HCCL_", "OMP_", "PYTORCH_NPU_", "VLLM_")):
        raise UnsupportedModeling("transport_or_allocator_unmodeled",
                                  "Capacity/algorithm or allocator effect has no complete binding.")
    return (_draft(item, "operational", None, value=_scrub(item.raw, secrets),
                   secrets=secrets),), ()


def _resolve(item, context, secrets):
    if item.kind == "visibility":
        if item.name == "visible_device_count":
            raise UnsupportedModeling("visible_device_count_insufficient", str(item.raw))
        raise UnsupportedModeling("visible_device_identity_unverified",
                                  "Visibility does not establish SKU/rank identity; "
                                  "supply explicit machine topology.")
    if item.kind == "wrapper":
        if item.name in {"--cpus", "--memory", "--shm-size", "--gpus"}:
            raise UnsupportedModeling("docker_wrapper_performance_unmodeled",
                                      "Wrapper performance control has no physical consumer.")
        return (_draft(item, "operational", None, secrets=secrets),), ()
    if item.kind == "env":
        return _resolve_env(item, context, secrets)
    name = item.name
    if name in _SCALAR:
        path, kind = _SCALAR[name]
        if item.raw is True or item.raw == "":
            raise BindingFailure(f"invalid_scalar:{name}")
        try:
            value = kind(item.raw)
        except (TypeError, ValueError) as exc:
            raise BindingFailure(f"invalid_scalar:{name}") from exc
        if kind is float and not math.isfinite(value):
            raise BindingFailure(f"invalid_scalar:{name}")
        return (_draft(item, "applied", "vllm.scalar", patches=((path, value),),
                       targets=(path,), value=value, secrets=secrets, key=path),), ()
    if name == "--enforce-eager":
        enabled = _bool_value(item, True)
        if not enabled:
            # The disabled spelling is the same canonical option: the later
            # occurrence must shadow an earlier enabling occurrence, so both
            # branches share ``enforce_eager`` as the duplicate-action key.
            return (_draft(item, "operational", None, value=False,
                           secrets=secrets, key="enforce_eager"),), ()
        patches = tuple(_eager_patches().items())
        return (_draft(item, "applied", "vllm.enforce_eager", patches=patches,
                       targets=tuple(path for path, _ in patches), value=True,
                       reason="Eager execution requires no ACLGraph capture.",
                       secrets=secrets, key="enforce_eager"),), ()
    if name in _BOOL:
        key, default, path = _BOOL[name]
        if name == "--enable-expert-parallel":
            enabled = _bool_value(item, default)
            if enabled and context["axes"]["pp"] > 1:
                raise UnsupportedModeling("expert_parallel_group_required",
                                          "PP>1 requires explicit expert group declarations.")
            value = context["axes"].get("ep", 1) if enabled else 1
            return (_draft(item, "applied", "vllm.expert_parallel_layout",
                           patches=(("/runtime/parallel/ep", value),),
                           targets=("/runtime/parallel/ep",), value=value,
                           secrets=secrets),), ()
        enabled = _bool_value(item, default)
        if key == "prefix_caching":
            if not enabled:
                requests = [dict(request, prefix_tokens=0) for request in
                            (context.get("base") or {}).get("workload", {}).get("requests", ())]
                patches = (("/workload/requests", requests),
                           ("/runtime/metadata/enable_prefix_caching", False))
                return (_draft(item, "applied", "vllm.prefix_caching", patches=patches,
                               targets=("/workload/requests",), effect="modeled",
                               reason="Disabled caching forces declared prefix hits to zero.",
                               value=False, secrets=secrets, key=key),), ()
            return (_draft(item, "applied", "vllm.prefix_caching",
                           targets=("/workload/requests",), effect="no_op",
                           reason="Declared prefix facts carry the hits; hit rate is not inferred.",
                           value=True, secrets=secrets, key=key),), ()
        return (_draft(item, "applied", "vllm.boolean", patches=((path, enabled),),
                       targets=(path,), value=enabled, secrets=secrets, key=key),), ()
    if name in _JSON_FLAGS:
        return _json(item, context, secrets)
    if name == "--all2all-backend":
        patches = _all2all(item, context)
        return (_draft(item, "applied", "vllm.moe_recipe", patches=patches,
                       targets=tuple(path for path, _ in patches), secrets=secrets),), ()
    if name in {"--quantization", "--dtype"}:
        raise UnsupportedModeling("checkpoint_metadata_required",
                                  "Backend choice only; supplied checkpoint quant/dtype metadata is required.")
    if is_secret_name(name):
        return (_draft(item, "operational", None, value="[REDACTED]", redacted=True,
                       secrets=secrets),), ()
    if name in _OPERATIONAL or not name.startswith("-"):
        return (_draft(item, "operational", None,
                       value=_scrub(item.raw, secrets), secrets=secrets),), ()
    raise UnsupportedModeling("cli_flag_without_consumer",
                              "Recognized CLI declaration has no physical consumer in this scope.")


__all__ = ["_compilation", "_disable_mlapo", "_eager_conflict", "_eager_patches",
           "_json", "_mlapo", "_resolve", "_resolve_env", "_speculative"]
