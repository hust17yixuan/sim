"""Regenerate FIX12 field-ownership evidence and the retained report.
[Role] Audit scoped files and render result.md from witnesses and pytest logs.
[Invariants] AST/byte checks and test failures gate reporting. Both GLM classes
  reconcile per stage; source payload arithmetic is distinct from hardware timing.
[Timeouts & Locks] Bounded CPU arithmetic and file reads; no network, NPU, or locks.
[Side-Effect Policy] Writes only requested report/artifact paths; imports write nothing.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import dataclasses as dc
import hashlib
import importlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace as NS

import nr_s7_fix3_verify as retained
from test_nr_s7_fix2_regressions_r2 import fix9_report, fix9_witnesses
from test_nr_s7_fix4_regressions import fix10_report, fix10_source_evidence, fix10_witnesses
from test_nr_s7_linear_kv_dspark_r1 import fix11_witnesses
from test_nr_s7_linear_kv_dspark_r2 import fix12_report, fix12_source_evidence, fix12_test_evidence
from test_nr_s7_fix3_regressions import fix11_report, fix11_source_evidence
from nr_s7_fix2_support import _case, _config, _dcp, _ds41, _glm, _kimi, _layer
from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.hybrid_kv_state_options import model_is_sparse_sfa
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state as E
from mindie_sim.linear_state_physics import compare_linear_state_memory
from mindie_sim.linear_state_replay import verify_commit_cost

ROOT = Path(__file__).resolve().parents[1]
REF_COMMIT = "6e3299ab492ddbadb4c8e3a00b88c982c9a28131"
SCOPED = retained.SCOPED
AUDITED = (*SCOPED, "mindie_sim/schema/model.py", "mindie_sim/prediction/structure.py")

# Independent domain oracle; omitted fields allow zero or have coupled constraints.
REQUIRED = {
    "RestoreExtent": "block_size hash_block_size",
    "RecoverSsmConfig": "pp dcp pcp tp kv_block_size",
    "GlmKdaExecution": "key_dim value_dim conv_kernel block_size mamba_block_size",
    "LinearLayer": "num_heads key_dim value_dim conv_kernel conv_width",
    "LayerPageGeometry": "tp spec_query_len",
    "LinearStateMemoryPlan": "sequence_tokens active_sequences page_stride_bytes",
    "VerifyCommitCost": "batch sequence_tokens hbm_bytes_per_s compute_flops_per_s ub_bytes conv_dim_stride accepted_tokens",
    "PrecopyCost": "layers batch hbm_bytes_per_s block_size",
    "PrefixReplayPlan": "isl",
    "CommitLaunchGeometry": "ub_bytes key_dim value_dim num_heads batch layers conv_dim_stride",
    "KvNzPlane": "head_size num_blocks num_kv_heads block_size",
    "KvNzLayout": "num_blocks num_kv_heads block_size nope_head_size",
    "Mte2ReadModel": "tokens logical_heads head_bytes cacheline_bytes pages_used nd_transactions nz_transactions nd_dma_ops nz_dma_ops useful_bytes nd_fetched_bytes nz_fetched_bytes",
    "NzReformatCost": "hbm_bytes_per_s",
    "KvSpecSignature": "block_size page_size_bytes",
    "DsparkSwaFootprint": "sequence_tokens swa_window_tokens kv_block_size layers row_bytes_per_token",
    "DcpDraftBlockCost": "draft_block_size dcp pcp tp num_q_heads q_nope_dim kv_heads head_dim latent_width num_draft_layers batch index_width_alignment index_entry_bytes index_length_bytes index_entries_per_us collective_bytes_per_s hbm_bytes_per_s compute_flops_per_s",
    "DSparkPrefixKeys": "pp_size tp_size block_size prefix_match_unit dcp_size",
    "DSparkPrefixPage": "shape0 shape1 numel element_size stride0_elements row_bytes",
    "DSparkPrefixPlan": "page_bytes page_stride_bytes block_size hash_block_size objects_per_key target_hidden_size num_blocks",
    "DSparkRegisteredPage": "scale size stride region_size",
    "DSparkPrefixRegistration": "num_blocks block_size hash_block_size page_bytes stride_bytes objects_per_key",
    "HybridKvStateOptions": "batch sequence_tokens conv_dim_stride hbm_bytes_per_s compute_flops_per_s",
    "BaselinePricing": "stages", "MemoryLedger": "pp logits_per_sequence",
    "DecodeCostModel": "hbm_bw cube_peak vector_peak layers hidden element_bytes context_tokens tp pp dp pcp dcp ep etp expert_dp kv_heads head_dim query_heads",
    "StepBreakdown": "batch_tokens active_replicas pp_microbatches straggler_ratio"}


@lru_cache(None)
def fix7_witnesses():
    """Replace real records across all 36 constructors, never forged objects."""
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.parallel_decode_cost import DecodeCostModel
    from mindie_sim.dspark_kv_physics import DeepseekV41DSparkSWACache
    from mindie_sim.memory_breakdown import build_memory_ledger
    from test_nr_s7_fix5_regressions import fix7_boundary_witnesses
    records, originals = {}, {}
    for filename in SCOPED[:26]:
        module = importlib.import_module(filename[:-3].replace("/", "."))
        for cls in vars(module).values():
            if inspect.isclass(cls) and cls.__module__ == module.__name__ and dc.is_dataclass(cls):
                originals[cls] = cls.__init__
                def capture(self, *args, _init=cls.__init__, **kwargs):
                    _init(self, *args, **kwargs)
                    records.setdefault(type(self).__name__, self)
                cls.__init__ = capture
    try:
        for name in ("_layout", "_linear", "_pool", "_gates", "_dspark", "_mcp", "_public", "_hardware", "_memory_edges"):
            getattr(retained, name)()
        DecodeCostModel.from_case(_case()).step(1)
        DeepseekV41DSparkSWACache()
    finally:
        for cls, init in originals.items():
            cls.__init__ = init
    assert set(records) == {cls.__name__ for cls in originals} and len(records) == 36
    rows = []
    for name, record in sorted(records.items()):
        for field in dc.fields(record):
            if type(getattr(record, field.name)) not in (int, float):
                continue
            for bad in (0, -1):
                required = field.name in REQUIRED.get(name, "").split()
                rejects = (bad < 0 and field.name != "kda_gate_lower_bound") or required
                try:
                    dc.replace(record, **{field.name: bad})
                except ModelingError:
                    outcome = "ModelingError"
                else:
                    outcome = "accepted"
                assert not rejects or outcome == "ModelingError", (name, field.name, bad)
                rows.append([name, field.name, bad, outcome])
    def reject(call, code=None):
        try:
            call()
        except ModelingError as error:
            assert code is None or code in str(error), str(error)
            return
        raise AssertionError("invalid record/container/mode accepted")
    enum_fields = {
        "RecoverSsmConfig": "mamba_cache_mode kv_role", "GlmKdaExecution": "cache_mode",
        "LinearLayer": "kind", "PrecopyCost": "mode", "KvNzPreconditions": "kv_role",
        "KvNzPlane": "name", "KvSpecSignature": "kind scope layout", "NzReformatCost": "path",
        "KimiK3DsparkGrouping": "mode", "HybridKvStateOptions": "mamba_cache_mode kv_role",
        "RecoverSsmValidation": "expected_cache_mode", "StepBreakdown": "scaling_mode overlap_mode"}
    enums = []
    for name, record in sorted(records.items()):
        for field in dc.fields(record):
            if field.name.endswith("dtype") or field.name in enum_fields.get(name, "").split():
                reject(lambda: dc.replace(record, **{field.name: "r7-invalid"}))
                enums.append([name, field.name])
    from mindie_sim.kv_pool_geometry import longest_run_recurrent_groups
    reject(lambda: longest_run_recurrent_groups(((0, "invalid"),)))
    reject(lambda: dc.replace(records["KvNzPreconditions"], enable_kv_nz=False, kv_role="invalid"))
    reject(lambda: dc.replace(records["RecoverSsmConfig"], use_replayssm=False, mamba_cache_mode="invalid"))
    case = _case()
    dense = dc.replace(records["DecodeCostModel"], moe_layers=0, top_k=0, num_experts=0,
                       expert_bytes=0.0, expert_params=0.0, launch_ms=0.0, l1_bytes=0.0, ub_bytes=0.0, tile_tokens=0)
    assert dense.step(1).step_ms > 0
    assert _glm(kernel_launch_ms=0.0).kernel_launch_ms == 0.0
    die = case.machine.dies[0]
    zero_link = dc.replace(die, rates=tuple((k, v) for k, v in die.rates if k != "link") + (("link", 0.0),))
    assert DecodeCostModel.from_case(dc.replace(case, machine=dc.replace(case.machine, dies=(zero_link, *case.machine.dies[1:])))).step(1).step_ms > 0
    linear = next(plan for plan in case.model.layer_plans if plan.attention == "kda")
    plans = tuple(dc.replace(linear, index=i) for i in range(case.model.layers))
    reject(lambda: E(dc.replace(case, model=dc.replace(case.model, layer_plans=plans)), _glm()), "mla_layer_plan_missing")
    reject(lambda: E(dc.replace(case, machine=dc.replace(case.machine, dies=()))))
    for capture in ("none", "full", "piecewise"):
        c = dc.replace(case, runtime=dc.replace(case.runtime, graph_capture_mode=capture))
        for mode in ("independent", "shared"):
            assert build_memory_ledger(c, graph_pool_mode=mode).graph_bytes >= 0
        reject(lambda: build_memory_ledger(dc.replace(c, runtime=dc.replace(c.runtime, metadata=(("graph_pool_mode", "invalid"),)))))
        reject(lambda: build_memory_ledger(c, moe_recipe="invalid"))
        reject(lambda: build_memory_ledger(c, moe_recipe=""))
    reject(lambda: build_memory_ledger(dc.replace(case, runtime=dc.replace(case.runtime, graph_capture_mode="invalid"))))
    return {"records": {"classes": len(records), "checks": len(rows), "rows": rows},
            "boundaries": fix7_boundary_witnesses(), "enum_fields": enums,
            "valid_controls": 9, "extra_rejections": 15}


def _exception_names(node):
    if node is None:
        return {"BaseException"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        return set().union(*(_exception_names(item) for item in node.elts))
    return set()


def audit_scope():
    files, timeout_violations, hygiene_violations = {}, [], []
    for name in AUDITED:
        data = (ROOT / name).read_bytes()
        text = data.decode()
        tree = ast.parse(text, filename=name)
        doc = ast.get_docstring(tree) or ""
        first = "\n".join(text.splitlines()[:30])
        row = {"lines": data.count(b"\n"), "cr_bytes": data.count(b"\r"),
               "trailing_lf": data.endswith(b"\n"), "broad_handlers": 0,
               "headers": all(marker in doc and marker in first for marker in retained.MARKERS)}
        if row["lines"] >= 700 or row["cr_bytes"] or not row["trailing_lf"] or not row["headers"]:
            hygiene_violations.append(name)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Try, ast.TryStar)):
                continue
            for index, handler in enumerate(node.handlers):
                if not _exception_names(handler.type) & {"Exception", "OSError", "BaseException"}:
                    continue
                row["broad_handlers"] += 1
                h = node.handlers[index - 1] if index else None
                guarded = (h is not None and _exception_names(h.type) == {"TimeoutError"} and len(h.body) == 1
                           and isinstance(h.body[0], ast.Raise) and h.body[0].exc is None)
                if not guarded:
                    timeout_violations.append(f"{name}:{handler.lineno}")
        files[name] = row
    caches = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
              if p.name in {"__pycache__", ".pytest_cache", ".ruff_cache"}
              or p.suffix in {".pyc", ".pyo", ".egg-info"}]
    workspace_cr = {str(p.relative_to(ROOT)): p.read_bytes().count(b"\r")
                    for p in ROOT.rglob("*") if p.is_file() and b"\r" in p.read_bytes()}
    all_python = sorted((len(p.read_bytes().splitlines()), str(p.relative_to(ROOT)))
                        for base in ("mindie_sim", "tests") for p in (ROOT / base).rglob("*.py"))
    return {"files": files, "timeout_violations": timeout_violations,
            "hygiene_violations": hygiene_violations, "workspace_caches": caches,
            "workspace_cr_bytes": sum(workspace_cr.values()), "workspace_cr_files": workspace_cr,
            "all_python_count": len(all_python),
            "outside_scope_at_least_700": [name for n, name in all_python if n >= 700 and name not in AUDITED]}


def _stage_kv(report, which):
    return next(c["per_stage_bytes"] for c in report["memory"][which]["components"]
                if c["name"] == "kv_and_state_pool_bytes")


def witnesses():
    data = {}
    retained.OUT.clear()
    for name in ("_layout", "_linear", "_pool", "_gates", "_dspark", "_mcp",
                 "_public", "_hardware", "_memory_edges"):
        getattr(retained, name)()
    data["retained"] = retained.OUT
    pools = []
    for label, pp, partition, recover in (("PP1 RecoverSSM", 1, None, True),
                                         ("PP2 standard (1,44)", 2, (1, 44), False),
                                         ("PP2 standard (22,23)", 2, (22, 23), False)):
        case = _case(pp=pp)
        if partition:
            case = dc.replace(case, runtime=dc.replace(case.runtime, pp_layer_partition=partition))
        report = E(case, _glm(use_replayssm=recover))
        pool = report["layout"]["mixed_pool"]
        for stage, record in enumerate(pool["stages"]):
            pools.append({"case": label, "stage": stage, "blocks": pool["blocks"],
                "main": record["main_class_bytes"], "small": record["small_class_bytes"],
                "small_baseline": pool["small_class_baseline_per_stage"][stage],
                "baseline": _stage_kv(report, "baseline")[stage],
                "delta": pool["delta_per_stage"][stage],
                "adjusted": _stage_kv(report, "adjusted")[stage],
                "missing": record["physical_pool_bytes"] - _stage_kv(report, "adjusted")[stage]})
    data["pools"] = pools
    data["commit"] = {label: verify_commit_cost(
        _config(mamba_cache_mode="align", enable_prefix_caching=True), [_layer()],
        batch=2, sequence_tokens=start, accepted_tokens=m, hbm_bytes_per_s=1e12)
        for label, start, m in (("normal m=5", 120, 5), ("boundary m=5", 126, 5),
                                ("normal m=3", 128, 3), ("boundary m=3", 127, 3))}
    data["cp"] = [{"topology": f"8/{pcp}/{dcp}", **DcpDraftBlockCost(
        **_dcp(pcp=pcp, dcp=dcp)).to_dict()} for pcp, dcp in ((1, 2), (2, 2), (2, 16))]
    cfg = _config(use_replayssm=False, speculative_k=8, model_type="generic_linear")
    data["generic"] = compare_linear_state_memory(cfg, [_layer()], sequence_tokens=2048)
    kimi = E(_case("kimi-k3"), _kimi(num_spec=8, dspark_active=False))
    data["kimi_k8"] = {"status": kimi["status"], "errors": kimi["errors"],
                       "comparison": kimi["linear_state"]["comparison"]}
    wrappers = (NS(hf_text_config=None, hf_config=NS(index_topk=0)),
                NS(hf_text_config=NS(), hf_config=NS(index_topk=8)),
                NS(hf_text_config=NS(index_topk=8), hf_config=NS(), index_kpool=1))
    data["sparse"] = [model_is_sparse_sfa(x) for x in wrappers]
    model = _case("deepseek-v3.2").model
    facts = {**dict(model.catalog_facts), "index_kpool": 1}
    data["sparse"].append(model_is_sparse_sfa(dc.replace(model, catalog_facts=tuple(facts.items()))))
    rejected = E(_case("deepseek-v4.1-flash"), _ds41(dspark_aux_layer_ids=(41,)))
    data["aux_rejected"] = {"errors": rejected["errors"],
        "memory": rejected["memory"]["aux_scratch_bytes"],
        "prefix": rejected["dspark"]["prefix"]["aux_scratch_bytes"],
        "stages": rejected["memory"]["aux_scratch_per_stage"]}
    from test_nr_s7_fix5_regressions import (
        boundary_matrix, sparse_witnesses, fix6_domain_witnesses, fix6_overflow_matrix)
    data["fix5"] = {"sparse": sparse_witnesses(), "validation": boundary_matrix()}
    data["fix6"] = {"domains": fix6_domain_witnesses(), "overflow": fix6_overflow_matrix()}
    data["fix7"] = {"domains": fix7_witnesses()}
    from test_nr_s7_fix4_regressions import fix8_workload_witnesses
    data["fix8"] = {"workload": fix8_workload_witnesses()}
    data["fix9"] = fix9_witnesses()
    data["fix10"] = {"permanent": fix10_witnesses()}
    data["fix11"] = {"permanent": fix11_witnesses()}
    return data

def protected_snapshot(artifact_dir):
    before = json.loads((artifact_dir / "before.json").read_text())
    counts, changed = {}, []
    for base, files in before.items():
        if base == str(ROOT):
            continue
        current = {}
        for path in Path(base).rglob("*"):
            if path.is_symlink():
                current[str(path.relative_to(base))] = {
                    "link": str(path.readlink()), "mtime_ns": path.lstat().st_mtime_ns}
            elif path.is_file():
                current[str(path.relative_to(base))] = [path.stat().st_size, path.stat().st_mtime_ns,
                                                       hashlib.sha256(path.read_bytes()).hexdigest()]
        counts[base] = len(current)
        if current != files:
            changed.append(base)
    return {"file_counts": counts, "changed_roots": changed}


def test_evidence(log, collection):
    text = log.read_text()
    summary = next(line for line in reversed(text.splitlines()) if re.search(r"\d+ passed", line))
    assert not re.search(r"\d+ (failed|error|skipped|xfailed|xpassed)", summary), summary
    passed = int(re.search(r"(\d+) passed", summary).group(1))
    counts = Counter(line.split("::")[0] for line in collection.read_text().splitlines()
                     if line.startswith("tests/") and "::" in line)
    assert sum(counts.values()) == passed, (passed, counts)
    return {"summary": summary, "passed": passed, "per_file": dict(sorted(counts.items())), "junit": fix12_test_evidence(log, counts, passed)}


def source_evidence(artifact_dir):
    probes = json.loads((artifact_dir / "r4_probe_results.json").read_text())
    assert probes["F04_small_class"]["missing_total_bytes"] == 0
    assert all(row["reference"] == row["local"] for row in probes["F03_sparse_source_predicate"])
    assert all(row["modeled_combine_bytes"] == row["source_packed_remote_bytes"]
               for row in probes["F08_source_merge"])
    commit = probes["F11_boundary_actual_tile"]
    assert commit["missing_boundary_record_bytes"] == commit["untimed_reported_bytes"] == 0
    assert probes["F09_correlated_registration_forgery"]["exception"] == "ModelingError"
    assert probes["F01_generic_standard_adapter"]["compare"]["accepted"]
    assert probes["F01_generic_standard_adapter"]["public_kimi_standard_k8"]["accepted"]
    assert probes["F06_rejected_aux_metadata"]["memory_aux_scratch_bytes"] == 0
    assert probes["F06_rejected_aux_metadata"]["prefix_aux_scratch_bytes"] == 0
    assert all(not row["accepted"] and row["exception"] == "ModelingError"
               for row in probes["F14_extended_validation"].values())
    matrices = json.loads((artifact_dir / "source_commit_matrix.json").read_text())
    assert [r["matrix_flops"] for r in matrices] == [3_145_728, 6_291_456]
    r5 = json.loads((artifact_dir / "r5_adversarial_results.json").read_text())
    sparse = r5["F03_wrapper_differential"]
    assert sparse["cases"] == 96 and sparse["mismatches"] == []
    assert all(r["reference"] == r["local"] for r in sparse["rows"])
    assert len(r5["F14_new_validation_counterexamples"]) == 8
    assert all(not r["accepted"] and r["exception"] == "ModelingError"
               for r in r5["F14_new_validation_counterexamples"].values())
    assert all(r["missing"] == 0 for r in r5["F04_source_allocation_grid"]["rows"])
    assert r5["F11_fresh_branch_pmf_grid"] == {"cases": 288, "failures": 0}
    assert r5["F08_fresh_packed_grid"]["cases"] == 63
    assert r5["F09_physical_registration"]["valid_scaled_and_bounds_cases"] == 9
    assert all(not r["accepted"] and r["exception"] == "ModelingError"
               for r in r5["F09_physical_registration"]["mutations"].values())
    assert all(r["bytes"] <= 250 and max(r["bpe"]) <= 250 for r in r5["F15_real_bpe_grid"]["rows"])
    r6 = json.loads((artifact_dir / "r6_boundary_results.json").read_text())
    expanded = r6["expanded_type_matrix"]
    assert not expanded["failures"] and not expanded["entered_original_body"]
    for row in r6["concrete_boundaries"].values():
        outcomes = [row] if "accepted" in row else [v for v in row.values() if isinstance(v, dict)]
        assert all(not r["accepted"] and r["type"] == "ModelingError" for r in outcomes)
    for name in ("r6_minimal_counterexamples", "r6_overflow_probe"):
        results = json.loads((artifact_dir / f"{name}.json").read_text())
        assert all(not r["accepted"] and r["exception"] == "ModelingError" for r in results.values())
    fresh = json.loads((artifact_dir / "r6_fresh_math_results.json").read_text())
    assert fresh["F03"] == {"extra_source_wrappers": 60, "normalized_catalog_cases": 90, "mismatches": 0}
    execution = json.loads((artifact_dir / "probe_execution.json").read_text())
    assert len(execution) == 10 and all(r["returncode"] == 0 for r in execution)
    r7 = json.loads((artifact_dir / "r7_fresh_results.json").read_text())
    for group in ("rate_domain", "decode_dimension_domain"):
        assert all(r[entry]["outcome"] == "ModelingError" for r in r7[group] for entry in ("constructor", "step"))
    assert all(r["outcome"] == "ModelingError" for r in r7["empty_workload"])
    assert all(r["outcome"] == ("ModelingError" if r["pool_mode"] == "r7-invalid" else "accepted") for r in r7["graph_disabled_enum"])
    assert r7["fresh_F03_source_wrapper_grid"] == {"comparisons": 108, "mismatches": 0}
    assert len(r7["rate_domain"]) == 12 and len(r7["decode_dimension_domain"]) == 16 and len(r7["empty_workload"]) == 4
    assert len(r7["nested_layer_prebranch"]) == 55 and all(r["outcome"] == "ModelingError" for r in r7["nested_layer_prebranch"])
    assert len(r7["fresh_F15_real_BPE"]["rows"]) == 12 and all(r["bytes"] <= 250 and max(r["tokens"]) <= 250 for r in r7["fresh_F15_real_BPE"]["rows"])
    all_probes = json.loads((artifact_dir / "all_probe_execution.json").read_text())
    assert len(all_probes) == 11 and all(r["returncode"] == 0 for r in all_probes)
    closure = json.loads((artifact_dir / "fix8_probe_execution.json").read_text())
    assert len(closure) == 12 and all(r["returncode"] == 0 for r in closure)
    r8 = json.loads((artifact_dir / "r8_fresh_results.json").read_text())
    assert all(len(r8[k]) == count and all(r["outcome"] == "ModelingError" for r in r8[k])
               for k, count in (("direct_workload", 6), ("replace_workload", 6), ("empty_consumers", 72)))
    r9, r9_fresh = (json.loads((artifact_dir / name).read_text()) for name in ("r9_malformed_results.json", "r9_fresh_results.json"))
    assert not r9["failures"] and r9["summary"]["rejection_percent"] == 100 and not r9_fresh["failures"]
    return {"r4": probes, "r5": r5, "matrix_source_execution": matrices,
            "r6_boundaries": r6, "r6_fresh": fresh,
            "execution": execution, "r9_malformed": r9, "r9_fresh": r9_fresh}


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "|" + "|".join("---" for _ in headers) + "|",
                       *("| " + " | ".join(str(v) for v in row) + " |" for row in rows)])


def _n(value):
    return f"{value:,.0f}"


def render(data, audit, tests, protected, artifact_dir):
    old = data["retained"]
    validation = data["fix5"]["validation"]
    sparse = data["fix5"]["sparse"]
    fix6 = data["fix6"]
    fix7 = data["fix7"]["domains"]
    expanded = data["source_probes"]["r6_boundaries"]["expanded_type_matrix"]
    groups = [k for k in data if k not in {"audit", "tests", "protected"}]
    validation_table = _table(["Production module", "Boundaries", "Typed parameters", "Invalid-type checks"],
        [[f"`{name}`", r["boundaries"], r["parameters"], r["checks"]]
         for name, r in validation["files"].items()])
    files = audit["files"]
    largest = max(files, key=lambda name: files[name]["lines"])
    implementations = [name for name in files if name.startswith("mindie_sim/")]
    largest_impl = max(implementations, key=lambda name: files[name]["lines"])
    pool_table = _table(["Case / stage", "IDs", "Main B", "Small B", "Small baseline B", "Baseline KV B", "Delta B", "Adjusted KV B", "Missing B"],
        [[f"{r['case']} / {r['stage']}", r["blocks"], *(_n(r[k]) for k in
          ("main", "small", "small_baseline", "baseline", "delta", "adjusted", "missing"))] for r in data["pools"]])
    commit_table = _table(["Trace", "Record reads B", "Commit matrix FLOPs", "Extra boundary matrix FLOPs", "Vector arithmetic FLOPs", "HBM B", "HBM-only ms"],
        [[label, *(_n(row[k]) for k in ("stage2_record_read_bytes", "stage2_matrix_flops",
          "boundary_matrix_flops", "stage2_vector_flops", "hbm_bytes")), f"{row['total_ms']:.9f}"]
         for label, row in ((label, data["commit"][label]) for label in
             ("normal m=5", "boundary m=5", "normal m=3", "boundary m=3"))])
    cp_table = _table(["TP/PCP/DCP", "Heads", "Packed shape", "Packed B", "TP remote B", "PCP remote B", "Combined remote B", "Attention FLOPs"],
        [[r["topology"], r["gathered_heads"], r["packed_merge_shape"], *(_n(r[k]) for k in
          ("packed_merge_bytes", "combine_tp_bytes", "combine_pcp_bytes", "combine_bytes", "attention_flops"))] for r in data["cp"]])
    public_table = _table(["Case", "Status", "Peak HBM B", "Net increment B", "Step ms", "Envelope bytes", "cl100k / o200k tokens"],
        [[r["label"], r["status"], _n(r["peak"]), _n(r["increment"]), f"{r['step']:.9f}", r["bytes"], r["bpe"]] for r in old["public"]])
    file_table = _table(["Scoped file", "Physical lines", "Broad handlers / guarded"],
        [[f"`{name}`", files[name]["lines"], f"{files[name]['broad_handlers']} / {files[name]['broad_handlers']}"] for name in files])
    test_table = _table(["Collected test file", "Passed"], tests["per_file"].items())
    closure_table = _table(["Finding", "Verified closure or retained scope"], [
        ("F01", "Generic and Kimi standard K=8 evaluate; RecoverSSM comparison applicable=False with reason/codes."),
        ("F02", "Public xPyD and typed PD routes retain the c8 scale plane."),
        ("F03", "Extracted source: 96 retained + 60 R6 + 108 R7 wrappers match; 90 fresh and 10 permanent normalized catalog cases pass."),
        ("F04", "Main + small page classes reconcile against the actual baseline on PP1 and both PP2 partitions; zero missing bytes."),
        ("F05", "Limiting rank, device budget, feasibility and subsequent signed delta tests pass."),
        ("F06", "Invalid auxiliary IDs report zero aggregate and per-stage scratch."),
        ("F07", "Noncausal split remains analytical; strict source mode rejects."),
        ("F08", "Packed gathered-head tensor drives TP all-to-all and full-tensor PCP gather."),
        ("F09", "Correlated registration forgery rejects from retained shape; scale, capacity, stride and inventory rechecked."),
        ("F10", "Unrounded accepted-length PMF retained; branch work integrated before expectation."),
        ("F11", "Branch-specific record reads; 2*N*(m+1) per accumulator; Vector work separate; A5/PR/strided grids retained."),
        ("F12", "Align fast-exit launches, none-mode zero work and separate prefix replay tests pass."),
        ("F13", "Typed NZ planes, multi-page/tail DMA, capacity and two-pass staging tests pass."),
        ("F14", f"R12-P2-01 field ownership, R11 authoritative contracts, R10 subclass/hashable-ID and R9/FIX8B regressions pass. {fix7['records']['classes']} retained record classes / {fix7['records']['checks']} scalar replacements; {validation['checks']} + {expanded['calls']} type, {fix6['domains']['checks']} domain and {fix6['overflow']['checks']} giant-integer checks pass."),
        ("F15", "Distinct error counts precede truncation; real-tokenizer and ASCII byte bounds pass."),
        ("G01", "All scoped broad handlers have a preceding TimeoutError re-raise; both generator handlers tested by injection."),
        ("F16", "This report is rendered from current witnesses, collection/run logs, and measured file counts.")])
    return f"""# NR-S7-FIX12 — Base-owned field domain validation closure

Reference: `/Users/maoxx241/code/vllm-ascend/vllm_ascend`, pinned commit `{REF_COMMIT}`.
`tests/nr_s7_fix4_verify.py` regenerated this document and `{artifact_dir}/result_numbers.json`.
These are CPU analytical witnesses, not measured NPU timings or hardware traffic counters.

Compatibility: `prediction/structure.py:parse_model` retains its immutable empty workload sentinel. Registered base fields retain their types, domains and recursive checks across subclass redeclarations. Request IDs must be nonblank, hashable strings. All ten default/explicit workload consumer forms reject the tested invalid records and base/subclass sentinels with ModelingError before implementation entry.

**R12 closure evidence and retained findings**

{closure_table}

{fix12_report(data)}{fix11_report(data)}
{fix10_report(data)}
{fix9_report(data)}

**R7 constructor, container and disabled-mode witnesses**

DecodeCostModel validates scalar counts/bytes and positive HBM/Cube/Vector supplies.
Required dimensions, including `hidden`, must be positive; dense MoE counts, absent
buffers and zero launch latency remain valid. Nested Device supplies reject nonpositive
values; an unequipped `rates.link=0` remains legal. The constructor sweep covers all
{fix7['records']['classes']} public dataclasses in the 26 modules: {fix7['records']['checks']}
zero/negative replacements, including output records. No scoped NamedTuples exist.
Signed gate bounds and optional zero work remain valid. Raw arithmetic errors cannot pass.

Permanent boundary rejections: `{fix7['boundaries']}`. Empty requests reject in
all four R7 call forms. Graph pool modes reject before capture selection, including
none/full/piecewise, aliases and empty strings. The index/enum audit also covers
empty dies, missing MLA plans, metadata pool modes and capture modes; the former
unbounded MLA `next` now raises ModelingError. There are {fix7['valid_controls']} valid controls
and {fix7['extra_rejections']} extra container/mode rejections; all {len(fix7['enum_fields'])} declared record enum/dtype fields reject invalid strings. All 12 retained probes pass.
The copied R7 source sweep adds 108 F03 comparisons and 12 real-BPE envelopes.

**Both GLM page classes, per PP stage**

Source `models/glm5next/cache_config.py:390,442` allocates
`global_blocks * (local_main_slots * main_page + local_small_slots * small_page)`.
Each stage subtracts the already-priced union of main components and
`index_k`/`index_pool_tail` once. A declared group spanning classes is charged
once; a larger declared baseline produces a signed release to the source allocation.
Other model families retain their existing conservative reconciliation policy.

{pool_table}

The TP8/K4/batch8/L2048 small baseline is 11,534,336 B of index keys plus
360,448 B of tail, totaling {_n(data['pools'][0]['small_baseline'])} B.
The added small-class difference is {_n(data['pools'][0]['small'] - data['pools'][0]['small_baseline'])} B.
`physical_pool_bytes` now includes both classes; `main_class_bytes` names the main-only allocation.
Standard PP2 is valid. RecoverSSM PP2 still rejects with `recoverssm_requires_pp1` and zero stage state ownership.

**Branch-specific commit work before PMF expectation**

Source `ops/triton/kda/recoverssm.py:577,603,610,787,796,1527` reads each full
BF16 key/gate vector per value program. Eight local 128x128 heads therefore read
`8 * (128//tile) * (2*128*2 + tile*4)` B per accepted token:
12,288 B at normal tile 64 and 20,480 B at boundary tile 32.
One accumulator executes `2*N*(m+1)` matrix FLOPs: m FMAs plus one
initial-state multiply/add after the loop. The boundary kernel executes two accumulators.

{commit_table}

Vector arithmetic is counted separately per head/value program: normalization
`3*K-1`, lower-bound gate `3*K`, weighted key/decay `2*K` per accumulator per token.
Exp, sigmoid and rsqrt counts are separate in the JSON; comparisons, selects,
conversions, metadata and address work are excluded. The declared FP32 rate prices
arithmetic FLOPs only; special operations have no supplied throughput rate.
Stage-1 work retains its existing analytical coefficient. Timing is a serial
HBM-plus-arithmetic-plus-launch/grid estimate, not a hardware latency guarantee.
With no compute/launch/grid cost, reported HBM time equals payload bytes / bandwidth;
the crossing m=5 witness has zero untimed reported payload bytes.

For K=4, alpha=0.8, E[M]={old['linear']['expectation']}; at start 2044 the
boundary probability is {old['linear']['pmf_cost']['probability']}. Each integer
trace selects its branch tile and prices reads/matrix/Vector work before weighting.
Both `Ascend950DT_9581` and `Ascend950PR_9579` retain commit tile 128,
boundary tile 64, 272 commit programs, 307 total programs and UB 253,952 B.
`conv_dim_stride=2` retains the 256-channel tile and 408 conv programs.

**Packed CP output tensor**

Source `attention/context_parallel/common_cp.py:222` and
`ops/triton/dcp/dcp_a2a.py:247,503` pack FP32 LSE into four BF16/FP16 elements
(one FP32 element for FP32 output). The BF16 row is `(512+4)*2 = 1,032 B`.
TP all-to-all preserves the full packed tensor extent and sends
`(scatter-1)/scatter` remotely; PCP all-gather receives `(pcp-1)*packed_bytes`.
Values below are per-rank one-direction remote payload bytes, excluding self traffic.

{cp_table}

The plain DCP row models the packed path; optional source fast-exchange dispatch
is not inferred. PCP always follows this packed composition. SWA extra_pairs stays zero.

**Prefix registration, applicability and typed validation**

Source `distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py:129`
drives registration: shape0 divisibility, scale=shape0/num_blocks,
scale*shape1=logical block, row capacity, kernel stride and scaled adjacency.
The region must equal `(shape0-1)*kernel_stride + kernel_bytes` from retained
physical facts. The loaded-page inventory must match every registered page.
R4's one-page-to-two-page dataclass forgery raises a typed shape-divisibility error.
Block-table coverage and hash-alignment checks remain enforced.

Generic standard K=8 remains {_n(data['generic']['standard']['total_bytes'])} B
for the one-layer witness, with comparison applicable={data['generic']['applicable']}.
Public Kimi standard K=8 returns `{data['kimi_k8']['status']}` with errors
`{data['kimi_k8']['errors']}`; its RecoverSSM counterfactual is inapplicable.
The four R4 sparse-predicate cases return `{data['sparse']}` in review order.
Invalid aux `(41,)` reports memory scratch `{data['aux_rejected']['memory']}`,
prefix scratch `{data['aux_rejected']['prefix']}`, stages `{data['aux_rejected']['stages']}`.
Malformed enum containers, missing/empty/wrong-length baseline vectors,
nonboolean conservative controls, and all six invalid staging types raise ModelingError.

F03 now distinguishes the normalized catalog ModelSpec from source HF wrappers.
The extracted pinned predicate matches **{sparse['r5_cases']}/{sparse['r5_cases']}**
SimpleNamespace/dataclass configurations, plus **{sparse['custom_cases']}/{sparse['custom_cases']}**
custom-class configurations: **{sparse['mismatches']} mismatches**. The permanent
suite also verifies {len(sparse['catalog'])} normalized catalog presence/default cases;
the retained explicit index_kpool values 0, 1 and None still exclude sparse SFA.

F14 now recursively validates supplied record fields after raw argument types and
before the decorated consumer body. This includes ModelSpec.layer_plans, every
LayerPlan's attention/consumed fields, nested MLA/draft records, Case containers
and typed stage vectors. Source-shaped HF wrappers keep their presence semantics.
The original R5 calls and copied R6 nested-layer/zero-byte/MLA counterexamples reject.

The concrete FIX6 sweep executes **{fix6['domains']['checks']} checks**:
`{fix6['domains']['families']}`. It covers both hardware architectures, disabled/draft
MLAPO, later malformed layers, zero/positive transfer bytes, zero/positive request
rate and latency, MLA/GQA dimensions, and overridden pool/renewal inputs.
Additional permanent tests cover disabled graph buckets, disabled/overridden EPLB
rank-vector lengths, parallel decode overrides,
none-mode overlap, nested Case/stage records, ledger stage/rank shapes and
computed-product overflow.

Numeric annotations and strict_int reject magnitudes beyond finite binary64 before
the body; strict_number guards conversion. Decorated functions, methods and property
getters translate OverflowError from computed products into ModelingError without
catching TimeoutError. Both signs of `10**400` produce **{fix6['overflow']['checks']}
typed rejections**, with no original body entered for the annotated numeric cases.
The inventory spans **{fix6['overflow']['boundaries']} boundaries in {fix6['overflow']['modules']}
modules**; boundaries without numeric parameters add no numeric mutation calls.

The systematic matrix executes **{validation['checks']} negative calls**. Its inventory
lists **{validation['boundaries']} public constructors/helpers/methods** and checks
**{validation['parameters']} typed parameters** in all **{validation['modules']}** scoped
production modules. It uses inert typed sentinels for other arguments and verifies
the rejected parameter's error, so a later attribute error cannot count as a pass.
The copied R6 expanded matrix adds **{expanded['calls']} type calls**, all rejected
before the original body. Valid inputs are exercised by the retained suite and
concrete FIX5/FIX6 controls.
Any-valued payloads retain their documented structural or sanitizing contracts;
these are finite executed sweeps, not a proof over arbitrary Python objects or all
possible numeric combinations. Structural mappings retain their explicit validators.

{validation_table}

**Retained physical witnesses and public evaluations**

Canonical TP8 KDA state {_n(old['linear']['state'])} B, allocated conv
{_n(old['linear']['conv_allocated'])} B, persistent records {_n(old['linear']['records'])} B,
page {_n(old['linear']['page'])} B; standard five slots {_n(old['linear']['standard_total'])} B,
align sixteen pages {_n(old['linear']['align_total'])} B.
The typed c8 page is {_n(old['layout']['pd_c8']['physical_page_bytes'])} B;
the 24-layer PD payload is {_n(old['layout']['pd_c8']['physical_payload_bytes'])} B.
The complete retained layout, replay, crossover, grouping, prefix, MLAPO, MCP,
hardware and public witness records are in `result_numbers.json`.

{public_table}

**Executed verification and scope**

`{tests['summary']}`. Collection matches {tests['passed']} executed tests.
Real tiktoken cl100k_base and o200k_base tests ran with the R2 package read-only
and tokenizer cache files copied into FIX12. Failures: 0; errors: 0; skips: 0.
The generator executed {len(groups)} witness groups: {', '.join(f'`{g}`' for g in groups)}.

{test_table}

Commands (run from `{artifact_dir}`; all temporary output remains there):

```sh
source {artifact_dir}/audit_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B {artifact_dir}/run_pytest.py
PYTHONDONTWRITEBYTECODE=1 python3 -B {ROOT}/tests/nr_s7_fix4_verify.py --artifact-dir {artifact_dir} --report {artifact_dir}/result.md
PYTHONDONTWRITEBYTECODE=1 python3 -B {ROOT}/tests/nr_s7_fix4_verify.py --artifact-dir {artifact_dir} --report {artifact_dir}/result.md --verify-only
```

The 37 retained scoped files plus schema/model.py and prediction/structure.py total {len(files)} files. All are strictly
below 700 physical lines, have all four headers in their module docstring and first
30 lines, zero CR bytes, and trailing LF. The AST scan finds zero unguarded broad handlers.
The largest scoped implementation is `{largest_impl}` ({files[largest_impl]['lines']} lines);
the largest scoped file is `{largest}` ({files[largest]['lines']} lines).
The full workspace scan finds {len(audit['workspace_caches'])} cache artifacts.
The entire workspace contains {audit['workspace_cr_bytes']} CR bytes.
The workspace contains {audit['all_python_count']} Python files; {len(audit['outside_scope_at_least_700'])} outside NR-S7 scope have 700+ lines.
The scoped line gate does not assert a repository-wide size limit.
{file_table}

Protected snapshots show {len(protected['changed_roots'])} changed roots (content, inventory, size and nanosecond mtime).
All 12 retained probes pass; `{artifact_dir}/copy_manifest.json` records adaptations.
Older labels are historical; probes are source shape/load/arithmetic doubles, not NPU kernels.
Tests and scope gates pass before reporting. `--verify-only` reruns {len(groups)} witness groups
and compares their canonical bytes, the complete JSON and report, writing only scratch evidence.

VERDICT: READY_FOR_INDEPENDENT_R13_SIGN_OFF
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    for mode in ("prepare-only", "verify-only"):
        parser.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in AUDITED}
    prepared = args.artifact_dir / "prepared_witnesses.json"
    if args.verify_only or args.prepare_only or not prepared.exists():
        data = witnesses()
        data["source_probes"] = source_evidence(args.artifact_dir)
        data["fix10"].update(fix10_source_evidence(args.artifact_dir))
        data["fix11"].update(fix11_source_evidence(args.artifact_dir))
        data["fix11"]["field_ownership"] = fix12_source_evidence(args.artifact_dir)
        data["fix7"]["source"] = json.loads((args.artifact_dir / "r7_fresh_results.json").read_text())
        data["fix8"]["source"] = json.loads((args.artifact_dir / "r8_fresh_results.json").read_text())
        if not args.verify_only:
            prepared.write_text(json.dumps({"source_hashes": hashes, "witnesses": data}, indent=2) + "\n")
        if args.prepare_only:
            print(str(prepared))
            return
    else:
        payload = json.loads(prepared.read_text())
        assert payload["source_hashes"] == hashes, "Source changed; regenerate with --prepare-only"
        data = payload["witnesses"]
    audit = audit_scope()
    tests = test_evidence(args.artifact_dir / "logs/pytest_full.log",
                          args.artifact_dir / "logs/collection.log")
    protected = protected_snapshot(args.artifact_dir)
    assert not audit["timeout_violations"] and not audit["hygiene_violations"]
    assert not audit["workspace_caches"] and not audit["workspace_cr_bytes"] and not protected["changed_roots"]
    data.update(audit=audit, tests=tests, protected=protected)
    text = render(data, audit, tests, protected, args.artifact_dir)
    assert "\r" not in text and text.endswith("\n")
    encoded = lambda value: (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    numbers = args.artifact_dir / "result_numbers.json"
    if args.verify_only:
        old = json.loads(numbers.read_text())
        equal = {k: encoded(data[k]) == encoded(old[k]) for k in data if k not in {"audit", "tests", "protected"}}
        evidence = dict(groups=equal, report_equal=text.encode() == args.report.read_bytes(), json_equal=encoded(data) == numbers.read_bytes())
        (args.artifact_dir / "readonly_equality.json").write_bytes(encoded(evidence))
        (args.artifact_dir / "result_regenerated_readonly.md").write_text(text)
        (args.artifact_dir / "result_numbers_regenerated.json").write_bytes(encoded(data))
        assert len(equal) == 16 and all(equal.values()) and evidence["report_equal"] and evidence["json_equal"], evidence
        print(json.dumps(evidence))
        return
    numbers.write_bytes(encoded(data))
    args.report.write_text(text)
    print(json.dumps({"report": str(args.report), "tests": tests["passed"], "scoped_files": len(AUDITED)}))


if __name__ == "__main__":
    main()
