"""S-owned service receipt invariants through the real public cohort path.

[Role] Prove ``simulate_service``/``run_cohort`` derive request timing from
  actual commit events on one Engine/Memory/Scheduler runtime, not from an
  isolated model duration or a second executor.
[Invariants] Every declared output token is committed; first token is the first
  committed event, finish is the last; terminal grants and live cache bytes are
  zero after the window; a caller-owned Execution is advanced in place.
[Side-Effect Policy] Remote CPU-only virtual engine; no devices or I/O.
"""
from __future__ import annotations

import pytest

from mindie_sim.prediction.api import run_cohort, simulate_service
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.structure import parse_case

from test_production_assembly_t14 import pipeline_case


def _commit_times(result):
    commits = {}
    for trace in result["traces"]:
        for request, when in dict(trace.get("declared_commit_ms") or {}).items():
            commits.setdefault(request, []).append(float(when))
        assert not trace.get("fallback_commit_ms")
    return commits


def test_service_request_timing_comes_from_actual_commit_events():
    result = simulate_service(pipeline_case(), event_limit=8, trace_limit=8)
    commits = _commit_times(result)
    assert sorted(commits) == ["a", "b"]
    for row in result["requests"]:
        times = commits[row["request_id"]]
        assert row["status"] == "served"
        assert row["generated_tokens"] == row["output_tokens"] == 2
        assert row["committed_tokens"] == 2
        assert row["first_token_abs_ms"] == pytest.approx(min(times))
        assert row["finish_ms"] == pytest.approx(max(times))
        assert row["e2e_ms"] == pytest.approx(row["finish_ms"] - row["arrival_ms"])
    assert result["production_entry"] == "simulate_service"
    assert result["metric_boundary"] == "device_token_commit"
    assert result["metrics"]["served_requests"] == 2


def test_service_window_drains_live_grants_and_cache_bytes():
    result = simulate_service(pipeline_case(), event_limit=8)
    memory = result["memory"]
    assert result["status"] == "modeled"
    assert result["execution"]["cohort_stop"] == "drained"
    assert result["execution"]["advance_calls"] >= 1
    assert isinstance(result["execution"]["runtime_uid"], str)
    assert result["execution"]["runtime_uid"]
    assert all(value == 0.0 for replica in memory[
        "live_grants_bytes_per_replica_stage"] for value in replica)
    assert all(value == 0.0 for replica in memory[
        "live_cache_bytes_per_replica_stage"] for value in replica)
    assert all(value == 0.0 for replica in memory[
        "physical_rank_reader_bytes_per_replica_stage"]
        for stage in replica for value in stage)


def test_run_cohort_advances_only_the_caller_runtime():
    raw = pipeline_case()
    parsed = parse_case(raw)
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory_policy=memory.policy, memory=memory)
    execution.engine.now = 0
    run = run_cohort(parsed, execution, [scheduler], memory, Frontend(),
                     trace_limit=0)
    assert run["engine"] is execution.engine
    assert run["stop_reason"] == "drained"
    assert scheduler.done is True
    assert run["serving_end"] == execution.engine.now
    assert execution.memory is memory
    assert execution.engine.memory is memory


def test_token_quota_is_a_window_not_a_shorter_request():
    raw = pipeline_case()
    result = simulate_service(
        raw, event_limit=4, token_quotas={"a": 1, "b": 1})
    assert result["status"] == "incomplete"
    assert result["execution"]["cohort_stop"] == "token_quota"
    for row in result["requests"]:
        assert row["output_tokens"] == 2
        assert row["generated_tokens"] <= 2
