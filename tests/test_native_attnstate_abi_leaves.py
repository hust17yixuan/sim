"""Independent-input contract for the A5 attention/index/state native leaves.

[Role] Exercise the standalone CPU path added to the state gather/scatter,
  indexer cache write and GDN q/k L2-norm leaves: a real native input
  (API + tensor descriptors + attrs + explicit implementation version) is
  normalized once, mapped to the existing family program and priced without a
  model case, weights or device initialization.

[Invariants] The state read/write order and slot are explicit and are never
  added as elapsed time.  Cache/token capacity comes from the supplied tensor
  shape, not from a captured sample.  A different legal shape follows the
  structural formula; an unknown ABI/attr/version or mismatched dtype/layout
  returns an explicit non-predicted status with a replayable input and no
  algorithm/envelope fallback.

[Timeouts & Locks] Pure CPU metadata/arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local inputs only; no product or global state.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from types import SimpleNamespace

def _close_number(value):
    """Numeric projection used only for tolerance comparisons of engine sums."""
    if isinstance(value, str):
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            return value
    return value


def _assert_close(left, right, *, rel=1e-9, path=""):
    """Same structure and service, allowing last-bit float accumulation.

    Engine clock/busy sums are floats: two callers that describe the same
    structural program can differ in the final decimal because they add the
    same amounts in a different order.  Service, work and structure remain an
    exact match; only the irrational float rounding is tolerated here.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), path
        for key in left:
            _assert_close(left[key], right[key], rel=rel, path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_close(a, b, rel=rel, path=f"{path}[{index}]")
        return
    a, b = _close_number(left), _close_number(right)
    if isinstance(a, bool) or isinstance(b, bool):
        assert a == b, path
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert a == pytest.approx(b, rel=rel, abs=1e-15), path
    else:
        assert a == b, path


from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.native_gdn import (
    STATE_EFFECT_ORDER,
    dynamic_gdn,
    state_effect,
)
from mindie_sim.prediction.piecewise.native_gather_ssm_states import (
    KERNEL_NAME as GATHER_ENTRY,
    R31_IMAGE_SHA,
    R31_REVISION,
    simulate_independent as simulate_gather,
)
from mindie_sim.prediction.piecewise.native_l2norm_fwd_kernel2 import (
    ENTRY as L2NORM_ENTRY,
    R31_IMAGE_SHA as L2NORM_IMAGE,
    REVISION as L2NORM_REVISION,
    simulate_independent as simulate_l2norm,
)
from mindie_sim.prediction.piecewise.native_r64_scatter_pa_cache import (
    CHILD_ENTRY as SCATTER_PA_ENTRY,
    ENTRY as SCATTER_PA_PARENT_ENTRY,
    R64_IMAGE_SHA,
    REVISION as R64_REVISION,
    simulate_independent as simulate_scatter_pa,
    simulate_parent as simulate_scatter_pa_parent,
)
from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
    ENTRY as ROPE_ENTRY,
    R64_IMAGE_SHA as ROPE_IMAGE,
    REVISION as ROPE_REVISION,
    simulate_independent as simulate_rope,
)
from mindie_sim.prediction.piecewise.native_scatter_ssm_states import (
    ENTRY as SCATTER_ENTRY,
    simulate_independent as simulate_scatter,
)


def _gather_request(*, batch=2, rows=2375, heads=24, dk=128, dv=128,
                    mask=(1, 0), state_dtype="fp32", index_dtype="int32",
                    extra_attrs=None, version=True):
    attrs = {
        "value_heads": heads, "dv": dv, "dk": dk,
        "state_layout": "V,K", "state_dtype": state_dtype,
        "index_dtype": index_dtype, "has_initial_state_values": list(mask),
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    request = {
        "api": {"namespace": "triton", "entry": GATHER_ENTRY},
        "tensors": {
            "state": {"shape": [rows, heads, dv, dk], "dtype": state_dtype},
            "indices": {"shape": [batch], "dtype": index_dtype},
            "has_initial_state": {"shape": [batch], "dtype": "bool"},
            "output": {"shape": [batch, heads, dv, dk],
                       "dtype": state_dtype},
        },
        "attrs": attrs,
    }
    if version:
        request["version"] = {"api": R31_REVISION, "source": R31_REVISION,
                              "image_id": R31_IMAGE_SHA}
    return request


def _scatter_request(*, batch=3, rows=2375, heads=24, dk=128, dv=128,
                     state_dtype="fp32", source_dtype="fp32",
                     index_dtype="int32", branch="fla_gdn_prefill",
                     extra_attrs=None, version=True):
    attrs = {
        "value_heads": heads, "dk": dk, "dv": dv,
        "state_layout": "V,K", "state_dtype": state_dtype,
        "index_dtype": index_dtype, "source_dtype": source_dtype,
        "scatter_branch": branch,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    request = {
        "api": {"namespace": "triton", "entry": SCATTER_ENTRY},
        "tensors": {
            "indices": {"shape": [batch], "dtype": index_dtype},
            "source": {"shape": [batch, heads, dv, dk],
                       "dtype": source_dtype},
            "state": {"shape": [rows, heads, dv, dk], "dtype": state_dtype},
        },
        "attrs": attrs,
    }
    if version:
        request["version"] = {"api": R31_REVISION, "source": R31_REVISION,
                              "image_id": R31_IMAGE_SHA}
    return request


def _scatter_pa_request(*, tokens=1024, rows=4096, dim=128, heads=32,
                        layout=2, block_size=128, kpool=1,
                        index_dtype="int32", extra_attrs=None,
                        version=True):
    attrs = {"index_head_dim": dim, "index_heads": heads, "layout": layout,
             "block_size": block_size, "kpool": kpool,
             "valid_rows_mode": "dynamic", "storage_dtype": "fp8_e4m3",
             "scale_storage_dtype": "fp32", "index_dtype": index_dtype}
    if extra_attrs:
        attrs.update(extra_attrs)
    request = {
        "api": {"namespace": "cann_kernel", "entry": SCATTER_PA_ENTRY},
        "tensors": {
            "update": {"shape": [tokens, 1, dim], "dtype": "fp8_e4m3"},
            "cache": {"shape": [rows, 1, 1, dim], "dtype": "fp8_e4m3"},
            "indices": {"shape": [tokens], "dtype": index_dtype},
            "scale_update": {"shape": [tokens, 1, 1], "dtype": "fp32"},
            "scale_cache": {"shape": [rows, 1, 1, 1], "dtype": "fp32"},
        },
        "attrs": attrs,
    }
    if version:
        request["version"] = {"api": R64_REVISION, "source": R64_REVISION,
                              "image_id": R64_IMAGE_SHA}
    return request


def _scatter_pa_parent_request(*, tokens=1024, rows=4096, dim=128, heads=32):
    return {
        "api": {"namespace": "torch", "entry": SCATTER_PA_PARENT_ENTRY},
        "version": {"api": R64_REVISION, "source": R64_REVISION,
                    "image_id": R64_IMAGE_SHA},
        "tensors": {
            "indexer_compress_cache": {
                "shape": [rows, 1, 1, dim + 4], "dtype": "uint8"},
            "x": {"shape": [tokens, 1, dim], "dtype": "bf16"},
            "slot_mapping": {"shape": [tokens], "dtype": "int32"},
        },
        "attrs": {"layout": 2, "index_head_dim": dim, "index_heads": heads,
                  "block_size": 128, "kpool": 1},
    }


def _l2norm_request(*, tokens=2, heads=8, dim=128, variant="q_l2norm",
                    extra_attrs=None, version=True):
    attrs = {"key_heads": heads, "head_dim": dim}
    if extra_attrs:
        attrs.update(extra_attrs)
    request = {
        "api": {"namespace": "triton", "entry": L2NORM_ENTRY},
        "conditions": {"variant": variant},
        "tensors": {
            "x": {"shape": [tokens, heads, dim], "dtype": "bf16"},
            "y": {"shape": [tokens, heads, dim], "dtype": "bf16"},
        },
        "attrs": attrs,
    }
    if version:
        request["version"] = {"api": L2NORM_REVISION,
                              "source": L2NORM_REVISION,
                              "image_id": L2NORM_IMAGE}
    return request


def _rope_request(*, tokens=32, heads=32, dim=128, rope_dim=64, role="query",
                  style=True, dtype="bf16", extra_attrs=None, variant=True):
    attrs = {"index_rope_role": role, "n_heads": heads, "head_dim": dim,
             "rope_dim": rope_dim, "is_neox_style": style,
             "rope_dtype": dtype}
    if extra_attrs:
        attrs.update(extra_attrs)
    request = {
        "api": {"namespace": "triton", "entry": ROPE_ENTRY},
        "tensors": {
            "qk": {"shape": [tokens, heads, dim], "dtype": dtype},
            "cos": {"shape": [tokens, rope_dim], "dtype": dtype},
            "sin": {"shape": [tokens, rope_dim], "dtype": dtype},
            "output": {"shape": [tokens, heads, dim], "dtype": dtype},
        },
        "attrs": attrs,
        "version": {"api": ROPE_REVISION, "source": ROPE_REVISION,
                    "image_id": ROPE_IMAGE},
    }
    if variant:
        request["conditions"] = {"variant": role}
    return request


def _duration(result):
    assert result["status"] == "predicted", result["reasons"]
    return Fraction(result["prediction"]["duration_ms"])


def test_state_effect_vocabulary_is_ordered_not_timed():
    assert STATE_EFFECT_ORDER == {"gather_read": 1, "chunk_compute": 2,
                                  "scatter_write": 3}
    effect = state_effect(order=1, direction="read", slot="s", tensor="t",
                          rows=2, dtype="fp32", role="read")
    assert "duration" not in effect and "ms" not in effect
    with pytest.raises(Exception):
        state_effect(order=1, direction="unknown", slot="s", tensor="t",
                     rows=2, dtype="fp32", role="read")


def test_gdn_chunk_prepare_declares_the_middle_state_effect_in_order():
    theta = SimpleNamespace(
        heads={"q": 8, "kv": 8, "dn": 128, "dv": 128},
        extras=(("chunk", 64),))
    operator = SimpleNamespace(name="gated_delta", theta=theta)
    compiled = SimpleNamespace(siblings=(operator,), operator=operator,
                               feature_metadata=())
    binding = SimpleNamespace(
        call_id="gdn-order-test",
        api={"aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd"},
        typed={"attrs": {"chunk_size": 64}, "dtype": "bf16",
               "layout": "BSND", "signature_hash": "sig", "shapes": {}},
        source={"sha256": "5e3b0fb0", "revision": "r31"}, unresolved=())
    geometry = WorkloadGeometry(
        (RequestGeometry(128, 0, phase="prefill"),), phase="prefill")
    prepared = dynamic_gdn(
        binding, compiled, geometry, validate=lambda _binding, actual: actual)
    assert prepared is not None
    effects = prepared["prepare"]["state_effects"]
    assert [item["order"] for item in effects] == [2, 2]
    assert [item["direction"] for item in effects] == ["read", "write"]
    assert [item["tensor"] for item in effects] == [
        "initial_state", "final_state"]
    assert effects[1]["requires_value"] is True
    sequence = prepared["prepare"]["state_effect_sequence"]["ordered"]
    assert [item["order"] for item in sequence] == [1, 2, 3]
    relation = prepared["prepare"]["binding_relation"]
    assert relation["relation"] == "one_to_many"
    assert relation["members_are_not_separate_timing_owners"] is True


# ---------------------------------------------------------------------------
# State gather
# ---------------------------------------------------------------------------

def test_gather_independent_input_prices_read_and_explicit_effect_order():
    result = simulate_gather(_gather_request())
    assert result["status"] == "predicted"
    effect = result["prediction"]["state_effects"][0]
    assert effect["order"] == 1
    assert effect["direction"] == "read"
    assert effect["slot"] == "ssm_state_cache"
    assert effect["rows"] == 1  # the real mask selects one of two sequences
    assert effect["cache_rows"] == 2375
    assert result["program"]["state_slot"] == "ssm_state_cache"
    assert "2375" not in result["program"]["formula"]
    assert result["binding"]["call_id"]
    assert result["occurrence"]["charge"] == "call"


def test_gather_new_shapes_follow_the_state_rows_formula():
    small = simulate_gather(_gather_request(batch=1, mask=(1,)))
    unseen = simulate_gather(_gather_request(
        batch=5, rows=9, heads=4, dk=64, dv=32, mask=(1, 1, 0, 1, 0)))
    derived = unseen["program"]["derived"]
    assert Fraction(derived["state_gather_rows"]) == 5
    assert Fraction(derived["state_gather_selected"]) == 3
    row_bytes = 4 * 64 * 32 * 4
    assert Fraction(derived["state_gather_read_bytes"]) == (
        3 * row_bytes + 5 * 5)  # selected rows + indices/mask per batch row
    assert Fraction(derived["state_gather_write_bytes"]) == 5 * row_bytes
    assert "2375" not in unseen["program"]["formula"]
    # Above the invocation floor the elapsed time follows the same work.
    many = simulate_gather(_gather_request(
        batch=32, heads=8, dk=128, dv=128, mask=(1,) * 32))
    few = simulate_gather(_gather_request(
        batch=8, heads=8, dk=128, dv=128, mask=(1,) * 8))
    assert _duration(many) > _duration(few)
    # A different cache capacity changes the descriptor, not the read work.
    other_capacity = simulate_gather(_gather_request(
        batch=5, rows=17, heads=4, dk=64, dv=32, mask=(1, 1, 0, 1, 0)))
    assert _duration(other_capacity) == _duration(unseen)
    assert other_capacity["prediction"]["state_effects"][0]["cache_rows"] == 17


def test_gather_rejects_unknown_abi_attr_and_capacity_guesses():
    unknown = simulate_gather(_gather_request(
        extra_attrs={"use_uniform_batch_default": True}))
    assert unknown["status"] == "no_match"
    assert unknown["fallback_performed"] is False
    assert any("unknown_attrs" in reason for reason in unknown["reasons"])
    bad_mask = simulate_gather(_gather_request(mask=(1,)))
    assert bad_mask["status"] == "no_match"
    assert "gather_abi_mask_values_mismatch" in bad_mask["reasons"]
    no_version = simulate_gather(_gather_request(version=False))
    assert no_version["status"] == "insufficient_fields"
    assert "version.image_id" in no_version["replay_minimal"]["missing"]
    request = _gather_request()
    request["version"]["image_id"] = "0" * 64
    assert simulate_gather(request)["status"] == "version_mismatch"


# ---------------------------------------------------------------------------
# State scatter
# ---------------------------------------------------------------------------

def test_scatter_independent_input_is_the_ordered_state_write():
    result = simulate_scatter(_scatter_request())
    assert result["status"] == "predicted"
    effect = result["prediction"]["state_effects"][0]
    assert effect["order"] == 3
    assert effect["direction"] == "write"
    assert effect["slot"] == "ssm_state_cache"
    assert effect["rows"] == 3
    assert effect["source_tensor"] == "source"
    assert result["program"]["state_slot"] == "ssm_state_cache"


def test_scatter_write_work_scales_with_selected_sequences():
    unseen = simulate_scatter(_scatter_request(
        batch=6, rows=8, heads=4, dk=64, dv=32))
    assert Fraction(
        unseen["program"]["derived"]["state_scatter_rows"]) == 6
    row_bytes = 4 * 64 * 32 * 4
    assert Fraction(
        unseen["program"]["derived"]["state_scatter_write_bytes"]) == 6 * row_bytes
    assert "2375" not in unseen["program"]["formula"]
    many = simulate_scatter(_scatter_request(
        batch=32, heads=8, dk=128, dv=128))
    few = simulate_scatter(_scatter_request(
        batch=8, heads=8, dk=128, dv=128))
    assert _duration(many) > _duration(few)


def test_scatter_rejects_uncast_source_and_unknown_branch():
    cast_needed = simulate_scatter(_scatter_request(source_dtype="fp16"))
    assert cast_needed["status"] == "no_match"
    assert "scatter_abi_source_cast_not_applied" in cast_needed["reasons"]
    unknown = simulate_scatter(_scatter_request(branch="unretained_branch"))
    assert unknown["status"] == "no_match"
    assert any("scatter_abi_branch_unknown" in reason
               for reason in unknown["reasons"])


# ---------------------------------------------------------------------------
# Indexer cache write (attention index state)
# ---------------------------------------------------------------------------

def test_index_cache_write_prices_two_child_kernels_and_keeps_capacity():
    result = simulate_scatter_pa(_scatter_pa_request())
    assert result["status"] == "predicted"
    assert len(result["occurrences"]) == 2
    assert [item["mode"] for item in result["occurrences"]] == [
        "indexer_k_cache", "indexer_scale_cache"]
    assert all(item["charge"] == "call" for item in result["occurrences"])
    effects = result["prediction"]["state_effects"]
    assert [item["order"] for item in effects] == [1, 2]
    assert [item["slot"] for item in effects] == [
        "indexer_k_cache", "indexer_scale_cache"]
    assert effects[0]["cache_rows"] == 4096
    assert effects[0]["cache_rows"] != 522240  # captured sample is evidence
    access = result["program"]["access"]
    assert access["boundary"] == "child_kernel"
    assert access["k_read_bytes"] == 1024 * 128 + 1024 * 4
    assert access["k_write_bytes"] == 1024 * 128
    assert access["scale_read_bytes"] == 1024 * 4 + 1024 * 4
    assert access["scale_write_bytes"] == 1024 * 4
    materialization = result["prediction"]["materialization"]
    assert Fraction(materialization["indexer_k_cache"]) == 1024 * 128
    assert Fraction(materialization["indexer_scale_cache"]) == 1024 * 4
    assert Fraction(materialization["cache_allocated"]) == 0
    assert result["program"]["parent_api"]["status"] == "declared_unimplemented"
    assert result["program"]["parent_api"][
        "child_tensors_are_not_parent_inputs"] is True


def test_index_cache_parent_api_is_a_declared_gap_not_child_inputs():
    gap = simulate_scatter_pa_parent(_scatter_pa_parent_request())
    assert gap["status"] == "not_implemented"
    assert "parent_child_input_mapping_not_retained" in gap["reasons"]
    assert gap["fallback_performed"] is False
    tensors = gap["replay_minimal"]["original_request"]["tensors"]
    assert set(tensors) == {"indexer_compress_cache", "x", "slot_mapping"}
    assert "update" not in tensors


def test_index_cache_binding_is_reusable_for_profiler_join():
    result = simulate_scatter_pa(
        _scatter_pa_request(),
        profiler_rows=[{"name": "ScatterPaCache"},
                       {"name": "unrelated_kernel"}])
    assert result["status"] == "predicted"
    join = result["profiler_join"]
    assert join["joined"] == 1 and join["unmatched"] == 1
    assert join["records"][0]["basis"] == "declared_child_kernel"
    assert sorted(join["records"][0]["call_ids"]) == sorted(
        item["call_id"] for item in result["occurrences"])


def test_index_cache_write_follows_new_token_and_cache_shapes():
    small = simulate_scatter_pa(_scatter_pa_request(tokens=1024))
    large = simulate_scatter_pa(_scatter_pa_request(
        tokens=2048, rows=8192, dim=64, heads=8))
    assert (Fraction(large["prediction"]["work"]["hbm"])
            > Fraction(small["prediction"]["work"]["hbm"]))
    assert large["program"]["access"]["k_write_bytes"] == 2048 * 64
    assert large["program"]["access"]["cache_capacity_rows"] == 8192


def test_index_cache_write_rejects_unknown_layout_and_kpool():
    layout = simulate_scatter_pa(_scatter_pa_request(layout=1))
    assert layout["status"] == "no_match"
    assert any("scatter_pa_abi_layout_unknown" in reason
               for reason in layout["reasons"])
    kpool = simulate_scatter_pa(_scatter_pa_request(kpool=4))
    assert kpool["status"] == "no_match"
    assert any("scatter_pa_abi_kpool_unsupported" in reason
               for reason in kpool["reasons"])
    dtype = simulate_scatter_pa(_scatter_pa_request(
        extra_attrs={"storage_dtype": "bf16"}))
    assert dtype["status"] == "no_match"
    assert any("leaf_tensor_attr_dtype_conflict" in reason
               for reason in dtype["reasons"])


# ---------------------------------------------------------------------------
# GDN q/k L2-norm
# ---------------------------------------------------------------------------

def test_l2norm_requires_the_real_q_or_k_variant():
    missing = simulate_l2norm(_l2norm_request())
    missing["conditions"] = {}
    request = _l2norm_request()
    request.pop("conditions")
    gap = simulate_l2norm(request)
    assert gap["status"] == "insufficient_fields"
    assert "conditions.variant" in gap["replay_minimal"]["missing"]
    query = simulate_l2norm(_l2norm_request(variant="q_l2norm"))
    key = simulate_l2norm(_l2norm_request(variant="k_l2norm"))
    assert query["status"] == key["status"] == "predicted"
    assert query["binding"]["api"]["entry"] == L2NORM_ENTRY
    assert query["binding"]["call_id"] != key["binding"]["call_id"]


def test_l2norm_row_mechanism_handles_unseen_tokens_and_dim():
    small = simulate_l2norm(_l2norm_request(tokens=8, heads=8, dim=128))
    large = simulate_l2norm(_l2norm_request(
        tokens=4096, heads=8, dim=128, variant="k_l2norm"))
    assert _duration(large) > _duration(small)
    unseen = simulate_l2norm(_l2norm_request(
        tokens=7, heads=4, dim=64, variant="k_l2norm"))
    assert unseen["program"]["flattened_shape"] == [28, 64]
    assert unseen["program"]["grid"] == [1]
    assert "key_heads" not in unseen["program"]["formula"]


def test_l2norm_output_dtype_changes_only_the_write_representation():
    base = _l2norm_request(tokens=8, heads=8, dim=128, variant="k_l2norm")
    fp32_request = _l2norm_request(tokens=8, heads=8, dim=128,
                                   variant="k_l2norm",
                                   extra_attrs={"output_dtype": "fp32"})
    fp32_request["tensors"]["y"]["dtype"] = "fp32"
    bf16 = simulate_l2norm(base)
    fp32 = simulate_l2norm(fp32_request)
    assert bf16["status"] == fp32["status"] == "predicted"
    elements = 8 * 128
    assert (Fraction(fp32["prediction"]["times_ms"]["hbm"])
            > Fraction(bf16["prediction"]["times_ms"]["hbm"]))
    mismatched = _l2norm_request(
        tokens=8, heads=8, dim=128, variant="k_l2norm",
        extra_attrs={"output_dtype": "fp32"})
    assert simulate_l2norm(mismatched)["status"] == "no_match"


def test_l2norm_unknown_attr_and_eps_default_are_explicit():
    unknown = simulate_l2norm(_l2norm_request(
        extra_attrs={"use_default_fla_norm": True}))
    assert unknown["status"] == "no_match"
    assert unknown["fallback_performed"] is False
    assert any("unknown_attrs" in reason for reason in unknown["reasons"])
    defaulted = simulate_l2norm(_l2norm_request())
    assert defaulted["occurrence"]["typed"]["attrs"]["eps_source"] == (
        "wrapper_default_1e-6")
    explicit = simulate_l2norm(_l2norm_request(extra_attrs={"eps": 1e-5}))
    assert explicit["occurrence"]["typed"]["attrs"]["eps_source"] == (
        "native_input")


# ---------------------------------------------------------------------------
# Index rotary (query and key are separate launches)
# ---------------------------------------------------------------------------

def test_rope_requires_the_real_query_or_key_launch():
    request = _rope_request()
    request.pop("conditions")
    missing = simulate_rope(request)
    assert missing["status"] == "insufficient_fields"
    assert "conditions.variant" in missing["replay_minimal"]["missing"]
    query = simulate_rope(_rope_request(role="query"))
    key = simulate_rope(_rope_request(role="key", heads=1))
    assert query["status"] == key["status"] == "predicted"
    assert query["binding"]["call_id"] != key["binding"]["call_id"]
    rotary = query["program"]["rotary"]
    assert rotary["dim"] == 64 and rotary["style"] == "neox"
    assert rotary["dtype"] == "bf16"
    assert rotary["head_dim"] == 128 and rotary["heads"] == 32
    assert rotary["modified_range"] == [0, 64]
    assert rotary["untouched_range"] == [64, 128]
    assert query["program"]["output_alias"] == {
        "output": "qk", "in_place": True}
    assert query["program"]["position_representation"] == (
        "selected_cos_sin_rows")


def test_rope_rotary_dim_drives_exact_access_and_math():
    narrow = simulate_rope(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"))
    wide = simulate_rope(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=128, role="query"))
    assert narrow["status"] == wide["status"] == "predicted"
    nper = narrow["program"]["access"]["per_token"]
    wper = wide["program"]["access"]["per_token"]
    assert nper == {"qk_read_bytes": 128, "cos_read_bytes": 64,
                    "sin_read_bytes": 64, "qk_write_bytes": 128,
                    "math_ops": 192, "conversion_elements": 192}
    assert wper == {"qk_read_bytes": 256, "cos_read_bytes": 128,
                    "sin_read_bytes": 128, "qk_write_bytes": 256,
                    "math_ops": 384, "conversion_elements": 384}
    assert narrow["program"]["access"]["effective_read_bytes"] == 32 * 256
    assert wide["program"]["access"]["effective_read_bytes"] == 32 * 512
    assert narrow["program"]["rotary"]["modified_range"] == [0, 64]
    assert narrow["program"]["rotary"]["untouched_range"] == [64, 128]
    assert (Fraction(narrow["prediction"]["work"]["vector"])
            == 32 * 192)
    assert (Fraction(wide["prediction"]["work"]["vector"])
            == 32 * 384)


def test_rope_per_value_dtype_and_conversion_elements():
    fp32_sin = _rope_request(tokens=32, heads=1, dim=128, rope_dim=64,
                             role="query")
    fp32_sin["tensors"]["cos"]["dtype"] = "fp32"
    fp32_sin["tensors"]["sin"]["dtype"] = "fp32"
    bf16_sin = _rope_request(tokens=32, heads=1, dim=128, rope_dim=64,
                             role="query")
    bf16_sin["tensors"]["cos"]["dtype"] = "fp32"
    bf16_sin["tensors"]["sin"]["dtype"] = "bf16"
    first = simulate_rope(fp32_sin)
    second = simulate_rope(bf16_sin)
    assert first["status"] == second["status"] == "predicted"
    assert first["program"]["access"]["per_token"] == {
        "qk_read_bytes": 128, "cos_read_bytes": 128,
        "sin_read_bytes": 128, "qk_write_bytes": 128,
        "math_ops": 192, "conversion_elements": 128}
    assert second["program"]["access"]["per_token"] == {
        "qk_read_bytes": 128, "cos_read_bytes": 128,
        "sin_read_bytes": 64, "qk_write_bytes": 128,
        "math_ops": 192, "conversion_elements": 160}
    assert first["program"]["access"]["per_token"][
        "qk_read_bytes"] + first["program"]["access"]["per_token"][
        "cos_read_bytes"] + first["program"]["access"]["per_token"][
        "sin_read_bytes"] == 384
    assert second["program"]["access"]["per_token"][
        "qk_read_bytes"] + second["program"]["access"]["per_token"][
        "cos_read_bytes"] + second["program"]["access"]["per_token"][
        "sin_read_bytes"] == 320
    assert first["program"]["cost_basis"] == (
        "family_native_rope_rdim_per_value")
    assert first["program"]["conversion"]["total_elements"] == 128 * 32
    assert second["program"]["conversion"]["total_elements"] == 160 * 32
    assert first["binding"]["typed"]["attrs"]["cos_dtype"] == "fp32"
    assert first["binding"]["typed"]["attrs"]["sin_dtype"] == "fp32"
    assert second["binding"]["typed"]["attrs"]["sin_dtype"] == "bf16"
    assert first["program"]["representation"]["conversion_separate"] is True
    assert Fraction(first["prediction"]["work"]["mte2"]) == 384 * 32
    assert Fraction(second["prediction"]["work"]["mte2"]) == 320 * 32
    assert Fraction(first["prediction"]["work"]["vector"]) == 192 * 32
    assert Fraction(second["prediction"]["work"]["vector"]) == 192 * 32
    all_bf16 = simulate_rope(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"))
    assert all_bf16["status"] == "predicted"
    assert all_bf16["program"]["conversion"]["total_elements"] == 192 * 32


def test_rope_new_tokens_and_style_follow_the_rotary_mechanism():
    small = simulate_rope(_rope_request(tokens=32))
    large = simulate_rope(_rope_request(tokens=4096, heads=8, dim=64,
                                        rope_dim=32, role="key"))
    assert (Fraction(large["prediction"]["work"]["vector"])
            > Fraction(small["prediction"]["work"]["vector"]))
    assert large["program"]["rotary"]["dim"] == 32
    assert large["program"]["rotary"]["style"] == "neox"
    gptj = simulate_rope(_rope_request(role="key", heads=1, style=False))
    assert gptj["program"]["rotary"]["style"] == "gptj"


def test_rope_rejects_role_and_rotary_guesses():
    mismatch = simulate_rope(_rope_request(
        role="query", extra_attrs={"index_rope_role": "key"}))
    assert mismatch["status"] == "no_match"
    assert "rope_abi_role_attr_mismatch" in mismatch["reasons"]
    unknown = simulate_rope(_rope_request(
        extra_attrs={"rope_implementation": "not_triton_siso"}))
    assert unknown["status"] == "no_match"
    assert any("unknown_attrs" in reason for reason in unknown["reasons"])
    bad_rope = simulate_rope(_rope_request(rope_dim=130, role="key", heads=1))
    assert bad_rope["status"] == "no_match"
    assert any("rope_abi_rotary_dims_invalid" in reason
               for reason in bad_rope["reasons"])


# ---------------------------------------------------------------------------
# P1-1 normalized descriptors, strides and replay fidelity
# ---------------------------------------------------------------------------

def test_gather_tensor_attr_dtype_conflict_is_rejected():
    request = _gather_request()
    request["tensors"]["state"]["dtype"] = "bf16"
    request["tensors"]["output"]["dtype"] = "bf16"
    result = simulate_gather(request)
    assert result["status"] == "no_match"
    assert any("leaf_tensor_attr_dtype_conflict:state:bf16!=fp32"
               in reason for reason in result["reasons"])
    assert result["fallback_performed"] is False


def test_gather_stride_is_preserved_and_consumed_in_address_span():
    request = _gather_request(batch=2, rows=5, heads=1, dk=128, dv=1)
    request["attrs"]["state_stride0"] = 160
    result = simulate_gather(request)
    assert result["status"] == "predicted"
    access = result["program"]["access"]
    assert access["state_row_payload_bytes"] == 128 * 4
    assert access["state_row_pitch_bytes"] == 160 * 4
    assert access["state_cache_span_bytes"] == 5 * 160 * 4
    assert access["state_read_bytes"] == 1 * 128 * 4
    assert result["occurrence"]["typed"]["attrs"]["state_stride0"] == 160


def test_scatter_mixed_source_state_dtype_uses_both_widths():
    result = simulate_scatter(_scatter_request(
        batch=1, heads=1, dk=64, dv=32, state_dtype="fp32",
        source_dtype="fp16", branch="use_fused_chunk"))
    assert result["status"] == "predicted"
    derived = result["prediction"]["derived"]
    assert Fraction(derived["source_read_bytes"]) == 4096
    assert Fraction(derived["index_read_bytes"]) == 4
    assert Fraction(derived["destination_write_bytes"]) == 8192
    assert Fraction(result["prediction"]["work"]["mte2"]) == 4100
    assert Fraction(result["prediction"]["work"]["mte3"]) == 8192
    assert result["program"]["access"][
        "source_destination_width_composed"] is False
    assert Fraction(result["prediction"]["stage_work"][0]["mte2"]) == 4100
    assert Fraction(result["prediction"]["stage_work"][0]["mte3"]) == 8192
    assert Fraction(result["prediction"]["stage_work"][0]["hbm"]) == 12292


def test_scatter_stride_and_launch_attrs_are_consumed_not_dropped():
    request = _scatter_request(batch=2, heads=1, dk=64, dv=32,
                               branch="use_fused_chunk")
    request["attrs"]["source_stride0"] = 2048 + 16
    request["attrs"]["indices_stride0"] = 2
    result = simulate_scatter(request)
    assert result["status"] == "predicted"
    access = result["program"]["access"]
    assert access["source_row_pitch_bytes"] == (2048 + 16) * 4
    assert access["index_stride_bytes"] == 2 * 4
    assert result["occurrence"]["typed"]["attrs"]["indices_stride0"] == 2
    bad = _scatter_request(batch=1, heads=1, dk=64, dv=32)
    bad["attrs"]["block_size"] = 512
    rejected = simulate_scatter(bad)
    assert rejected["status"] == "no_match"
    assert any("leaf_fixed_launch_attr_mismatch:block_size"
               in reason for reason in rejected["reasons"])


def test_rejected_request_replay_preserves_unknown_fields_and_semantics():
    request = _gather_request()
    request["attrs"]["unknown_semantic_attr"] = "must_stay_visible"
    result = simulate_gather(request)
    replay = result["replay_minimal"]
    assert result["status"] == "no_match"
    assert replay["replayable"] is False
    assert replay["original_request"]["attrs"][
        "unknown_semantic_attr"] == "must_stay_visible"
    assert "unknown_semantic_attr" in replay["rejected"]["attrs"]
    assert "declared_template_is_not_a_prediction_request" in replay[
        "semantics"]
    assert replay["declared_template"]["attrs"].get(
        "unknown_semantic_attr") is None


def test_rope_output_alias_dtype_conflict_is_rejected():
    request = _rope_request(role="key", heads=1)
    request["tensors"]["output"]["dtype"] = "fp32"
    result = simulate_rope(request)
    assert result["status"] == "no_match"
    assert any("leaf_alias_dtype_conflict:output"
               in reason for reason in result["reasons"])


# ---------------------------------------------------------------------------
# P1-3 L2Norm real row/BD/tail/capacity geometry
# ---------------------------------------------------------------------------

def _l2norm_rows_request(*, rows, head_dim, output_dtype="bf16",
                         tokens=None, heads=None):
    request = _l2norm_request(
        tokens=tokens or 1, heads=heads or rows, dim=head_dim,
        variant="q_l2norm")
    if output_dtype != "bf16":
        request["attrs"]["output_dtype"] = output_dtype
        request["tensors"]["y"]["dtype"] = output_dtype
    if tokens is None:
        # Use an explicit [M,N] request so the row axis is unambiguous.
        request["tensors"]["x"] = {"shape": [rows, head_dim],
                                   "dtype": "bf16"}
        request["tensors"]["y"] = {"shape": [rows, head_dim],
                                   "dtype": output_dtype}
    return request


def test_l2norm_same_elements_different_rows_and_bd_change_issued_work():
    rows_64 = simulate_l2norm(_l2norm_rows_request(rows=8, head_dim=64))
    rows_128 = simulate_l2norm(_l2norm_rows_request(rows=4, head_dim=128))
    assert rows_64["status"] == rows_128["status"] == "predicted"
    a = rows_64["program"]["access"]
    b = rows_128["program"]["access"]
    assert a["M"] * a["N"] == b["M"] * b["N"] == 512
    assert a["BD"] == 64 and b["BD"] == 128
    assert a["tile_rows"] == b["tile_rows"] == 32
    assert a["tail_rows"] == 24 and b["tail_rows"] == 28
    assert a["issued_read_bytes"] == 8 * 64 * 2
    assert b["issued_read_bytes"] == 4 * 128 * 2
    assert a["issued_write_bytes"] == 8 * 64 * 2
    assert b["issued_write_bytes"] == 4 * 128 * 2
    assert a["compute_math_ops"] == 32 * 64 * 5
    assert b["compute_math_ops"] == 32 * 128 * 5
    assert a["compute_tile_rows"] == b["compute_tile_rows"] == 32


def test_l2norm_capacity_boundary_and_fp32_output_representation():
    at_limit = simulate_l2norm(_l2norm_rows_request(rows=2, head_dim=32768))
    assert at_limit["status"] == "predicted"
    over = simulate_l2norm(_l2norm_rows_request(rows=2, head_dim=32769))
    assert over["status"] == "constraint_violation"
    assert any("l2norm_feature_dim_exceeds_wrapper_limit" in reason
               for reason in over["reasons"])
    fp32 = simulate_l2norm(
        _l2norm_rows_request(rows=8, head_dim=64, output_dtype="fp32"))
    assert fp32["status"] == "predicted"
    access = fp32["program"]["access"]
    assert access["issued_write_bytes"] == 8 * 64 * 4
    assert access["effective_write_bytes"] == 8 * 64 * 4
    assert access["compute_math_ops"] == 32 * 64 * 5
    assert fp32["program"]["reduction"]["capacity_basis"] == (
        "65536 // element_size")


# ---------------------------------------------------------------------------
# Remaining native entries: Cat, MulsAdd, LayerNorm, TBMM, GDN
# ---------------------------------------------------------------------------

def _cat_request(*, tokens=3, heads=4, latent=8, rope=2):
    from mindie_sim.prediction.piecewise.native_r64_aclnn_cat import (
        ENTRY, R64_IMAGE_SHA, REVISION)
    return {
        "api": {"namespace": "torch", "entry": ENTRY},
        "version": {"api": REVISION, "source": "r64",
                    "image_id": R64_IMAGE_SHA},
        "tensors": {
            "ql_nope": {"shape": [tokens, heads, latent], "dtype": "bf16"},
            "q_pe": {"shape": [tokens, heads, rope], "dtype": "bf16"},
            "query": {"shape": [tokens, heads, latent + rope],
                      "dtype": "bf16"},
        },
        "attrs": {"concat_dim": -1},
    }


def _muls_request(*, tokens=5, hidden=16, scale=2.5, dtype="bf16"):
    from mindie_sim.prediction.piecewise.native_r64_muls_add import (
        ENTRY, R64_IMAGE_SHA, R64_VERSION)
    return {
        "api": {"namespace": "torch", "entry": ENTRY},
        "version": {"api": R64_VERSION, "source": "r64",
                    "image_id": R64_IMAGE_SHA},
        "tensors": {
            "x": {"shape": [tokens, hidden], "dtype": dtype},
            "y": {"shape": [tokens, hidden], "dtype": dtype},
            "output": {"shape": [tokens, hidden], "dtype": dtype},
        },
        "attrs": {"hidden": hidden, "scale": scale, "dtype": dtype,
                  "act_bytes": 2.0, "ops_per_element": 2,
                  "read_bytes_per_token": 2 * hidden * 2.0,
                  "write_bytes_per_token": hidden * 2.0,
                  "combine_variant": "muls_add", "scale_owner": "join",
                  "token_domain": "prefill"},
    }


def _layernorm_request(*, rows=4, width=64, impl_mode=None):
    from mindie_sim.prediction.piecewise.native_r64_layer_norm_with_impl_mode import (
        ENTRY, R64_IMAGE_SHA, REVISION)
    attrs = {"index_dim": width, "eps": 1e-5, "dtype": "bf16"}
    if impl_mode is not None:
        attrs["impl_mode"] = impl_mode
    return {
        "api": {"namespace": "torch", "entry": ENTRY},
        "version": {"api": REVISION, "source": "r64",
                    "image_id": R64_IMAGE_SHA},
        "tensors": {
            "input": {"shape": [rows, width], "dtype": "bf16"},
            "gamma": {"shape": [width], "dtype": "bf16"},
            "beta": {"shape": [width], "dtype": "bf16"},
            "output": {"shape": [rows, width], "dtype": "bf16"},
            "mean": {"shape": [rows, 1], "dtype": "fp32"},
            "rstd": {"shape": [rows, 1], "dtype": "fp32"},
        },
        "attrs": attrs,
    }


def _tbmm_request(*, tokens=2, batch=4, k=8, n=6, perm=(1, 0, 2)):
    from mindie_sim.prediction.piecewise.native_r64_transpose_batch_matmul import (
        ENTRY, R64_IMAGE_SHA, REVISION)
    return {
        "api": {"namespace": "torch", "entry": ENTRY},
        "version": {"api": REVISION, "source": "r64",
                    "image_id": R64_IMAGE_SHA},
        "tensors": {
            "input": {"shape": [tokens, batch, k], "dtype": "bf16"},
            "weight": {"shape": [batch, k, n], "dtype": "bf16"},
            "output": {"shape": [tokens, batch, n], "dtype": "bf16"},
        },
        "attrs": {"matrix_batch": batch, "matrix_k": k, "matrix_n": n,
                  "perm_x1": list(perm), "perm_y": list(perm)},
    }


def _gdn_request(*, lengths=(64, 64), physical_batch=1,
                 physical_axis_tokens=None, include_lengths=True,
                 qsl=None):
    from mindie_sim.prediction.piecewise.native_gdn import (
        GDN_ABI_ENTRY, GDN_ABI_SOURCE_REVISION)
    from mindie_sim.prediction.piecewise.native_scatter_ssm_states import (
        QWEN36_IMAGE_SHA)
    sequences, tokens = len(lengths), sum(lengths)
    axis = tokens if physical_axis_tokens is None else int(physical_axis_tokens)
    attrs = {"chunk_size": 64, "scale": 128 ** -0.5,
             "use_qk_l2norm_in_kernel": True, "use_exp2": True,
             "disable_recompute": True, "state_v_first": True,
             "output_final_state": True,
             "use_gate_in_kernel": False,
             "use_beta_sigmoid_in_kernel": False,
             "allow_neg_eigval": False,
             "return_intermediate_states": False}
    if include_lengths:
        attrs["lengths"] = list(lengths)
    if qsl is not None:
        attrs["cu_seqlens"] = list(qsl)
    return {
        "api": {"namespace": "aclnn", "entry": GDN_ABI_ENTRY,
                "aclnn_symbol": GDN_ABI_ENTRY},
        "version": {"api": GDN_ABI_SOURCE_REVISION,
                    "source": GDN_ABI_SOURCE_REVISION,
                    "image_id": QWEN36_IMAGE_SHA},
        "tensors": {
            "q": {"shape": [physical_batch, axis, 8, 128], "dtype": "bf16"},
            "k": {"shape": [physical_batch, axis, 8, 128], "dtype": "bf16"},
            "v": {"shape": [physical_batch, axis, 8, 128], "dtype": "bf16"},
            "g": {"shape": [physical_batch, axis, 8], "dtype": "fp32"},
            "beta": {"shape": [physical_batch, axis, 8], "dtype": "bf16"},
            "initial_state": {"shape": [sequences, 8, 128, 128],
                              "dtype": "fp32"},
            "o": {"shape": [physical_batch, axis, 8, 128], "dtype": "bf16"},
            "final_state": {"shape": [sequences, 8, 128, 128],
                            "dtype": "fp32"},
        },
        "attrs": attrs,
    }


def test_cat_independent_entry_preserves_last_axis_value():
    from mindie_sim.prediction.piecewise.native_r64_aclnn_cat import (
        simulate_independent)
    result = simulate_independent(_cat_request())
    assert result["status"] == "predicted"
    assert result["program"]["concat_axis"] == -1
    assert result["program"]["access"]["read_bytes"] == 3 * 4 * 10 * 2
    assert result["program"]["access"]["write_bytes"] == 3 * 4 * 10 * 2
    assert result["occurrence"]["typed"]["shapes"]["query"] == [3, 4, 10]


def test_muls_add_independent_entry_refuses_identity_or_router_scale():
    from mindie_sim.prediction.piecewise.native_r64_muls_add import (
        simulate_independent)
    result = simulate_independent(_muls_request())
    assert result["status"] == "predicted"
    assert result["program"]["variant"] == "muls_add"
    assert result["program"]["scale"] == 2.5
    assert result["program"]["double_scaling"] is False
    identity = _muls_request(scale=1.0)
    rejected = simulate_independent(identity)
    assert rejected["status"] == "no_match"
    assert any("muls_add_identity_scale_is_plain_add" in reason
               for reason in rejected["reasons"])
    router = _muls_request()
    router["attrs"]["scale_owner"] = "router"
    rejected = simulate_independent(router)
    assert rejected["status"] == "no_match"
    assert any("muls_add_scale_owned_by_router" in reason
               for reason in rejected["reasons"])


def test_layernorm_independent_entry_keeps_impl_mode_gap():
    from mindie_sim.prediction.piecewise.native_r64_layer_norm_with_impl_mode import (
        simulate_independent)
    result = simulate_independent(_layernorm_request())
    assert result["status"] == "predicted"
    assert result["program"]["rows"] == 4
    assert result["program"]["index_dim"] == 64
    assert result["program"]["impl_mode"] is None
    assert result["program"]["outputs"] == ["output", "mean", "rstd"]
    declared = simulate_independent(_layernorm_request(impl_mode=1))
    assert declared["status"] == "no_match"
    assert any("layernorm_impl_mode_value_unresolved" in reason
               for reason in declared["reasons"])


def test_transpose_batchmatmul_independent_entry_keeps_permutation_and_work():
    from mindie_sim.prediction.piecewise.native_r64_transpose_batch_matmul import (
        simulate_independent)
    result = simulate_independent(_tbmm_request())
    assert result["status"] == "predicted"
    assert result["program"]["matrix"] == {"batch": 4, "tokens": 2,
                                           "k": 8, "n": 6}
    assert result["program"]["perm_x1"] == [1, 0, 2]
    assert result["program"]["access"]["weight_payload_bytes"] == 4 * 8 * 6 * 2
    assert result["program"]["layout_status"] == "logical_view_only"
    assert result["program"]["physical_native_layout_selected"] is False
    wrong = simulate_independent(_tbmm_request(perm=(0, 1, 2)))
    assert wrong["status"] == "no_match"
    assert any("tbmm_abi_perm_x1_unsupported" in reason
               for reason in wrong["reasons"])


def test_gdn_independent_entry_preserves_packed_partition():
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    even = simulate_independent(_gdn_request(lengths=(64, 64)))
    assert even["status"] == "predicted"
    partition = even["program"]["partition"]
    assert partition["physical_batch"] == 1
    assert partition["sequence_count"] == 2
    assert partition["cu_seqlens"] == [0, 64, 128]
    assert partition["chunks"] == 2
    assert partition["chunk_head_tasks"] == 16
    assert partition["state_rows"] == 2
    assert even["program"]["call_report"]["execution"]["recipes"]
    assert even["program"]["call_report"]["operator_occurrences"] == 18
    assert [item["order"] for item in even["prediction"]["state_effects"]] == [
        2, 2]
    ragged = simulate_independent(_gdn_request(lengths=(1, 127)))
    assert ragged["status"] == "predicted"
    partition = ragged["program"]["partition"]
    assert partition["cu_seqlens"] == [0, 1, 128]
    assert partition["chunks"] == 3
    assert partition["chunk_head_tasks"] == 24
    assert partition["state_rows"] == 2
    ports = ragged["program"]["state_ports"]
    assert [item["queries"] for item in ports] == [1, 127]
    assert [item["read_version"] for item in ports] == [0, 0]
    assert [item["write_version"] for item in ports] == [1, 127]
    assert all(item["last_use"] == "call_return_then_scatter"
               for item in ports)
    assert ragged["program"]["state_value"]["final"]["shape"] == [
        2, 8, 128, 128]
    missing = _gdn_request(lengths=(64, 64))
    missing["attrs"].pop("lengths")
    gap = simulate_independent(missing)
    assert gap["status"] == "no_match"
    assert any("sequence_partition_required" in reason
               for reason in gap["reasons"])


def _split_request(*, implementation=None):
    from mindie_sim.prediction.piecewise.native_split_qkv_rmsnorm_mrope import (
        ENTRY, R31_IMAGE_SHA, SOURCE)
    attrs = {"q_hidden_size": 32, "kv_hidden_size": 8,
             "head_dim": 8, "q_size": 32, "kv_size": 8, "gate_size": 0,
             "rope_dim": 8, "eps": 1e-6,
             "mrope_section": [2, 1, 1], "mrope_interleaved": True}
    if implementation is not None:
        attrs["implementation"] = implementation
    return {
        "api": {"namespace": "cann_kernel", "entry": ENTRY},
        "version": {"api": "r31-original-capture",
                    "source": SOURCE["revision"], "image_id": R31_IMAGE_SHA},
        "tensors": {
            "input": {"shape": [4, 48], "dtype": "bf16"},
            "q_weight": {"shape": [8], "dtype": "bf16"},
            "k_weight": {"shape": [8], "dtype": "bf16"},
            "cos_sin_cache": {"shape": [3, 4, 8], "dtype": "bf16"},
            "positions": {"shape": [3, 4], "dtype": "int64"},
            "q": {"shape": [4, 32], "dtype": "bf16"},
        },
        "attrs": attrs,
    }


def test_split_qkv_mrope_is_partial_and_rejects_unknown_implementation():
    from mindie_sim.prediction.piecewise.native_split_qkv_rmsnorm_mrope import (
        simulate_independent)
    result = simulate_independent(_split_request())
    assert result["status"] == "partial"
    assert result["native_call_succeeded"] is False
    assert result["program"]["complete"] is False
    assert result["occurrence"]["charge"] == "unresolved"
    assert result["prediction"]["scoring"] == "excluded_partial"
    assert result["program"]["known_outputs"] == ["q"]
    assert result["program"]["unverified_outputs"] == ["k", "v", "gate"]
    assert result["program"]["positions"]["semantics"] == "unverified"
    assert "split_k_v_gate_outputs_unrecovered" in result["reasons"]
    bindings = result["program"]["output_bindings"]
    assert bindings["q"]["verified"] is True
    assert bindings["k_candidate"]["verified"] is False
    assert bindings["v_candidate"]["verified"] is False
    assert bindings["gate_candidate"]["verified"] is False
    assert result["program"]["positions_binding"]["semantics"] == "unverified"
    assert result["program"]["implementation_selection"]["selected"] == (
        "observed_r31_mrope_kernel")
    unknown = simulate_independent(_split_request(
        implementation="generic_elementwise_upper_bound"))
    assert unknown["status"] == "no_match"
    assert any("split_qkv_implementation_not_declared" in reason
               for reason in unknown["reasons"])
    assert unknown["replay_minimal"]["original_request"]["attrs"][
        "implementation"] == "generic_elementwise_upper_bound"
    assert unknown["replay_minimal"]["replayable"] is False


def test_leaf_catalog_advertises_all_entries_for_t12():
    from mindie_sim.prediction.piecewise.native_gdn import leaf_abi_specs
    specs = leaf_abi_specs()
    entries = {spec.entry for spec in specs}
    assert len(specs) == 11
    assert "aclnnChunkGatedDeltaRuleFwd" in entries
    assert "torch.cat" in entries
    assert "ScatterPaCache" in entries
    assert "split_qkv_rmsnorm_mrope_kernel" in entries
    assert all(spec.to_dict()["entry"] for spec in specs)


def test_cost_serialization_and_stage_work_for_three_program_paths():
    import json
    mixed = simulate_scatter(_scatter_request(
        batch=1, heads=1, dk=64, dv=32, state_dtype="fp32",
        source_dtype="fp16", branch="use_fused_chunk"))
    fp32 = simulate_l2norm(_l2norm_rows_request(
        rows=8, head_dim=64, output_dtype="fp32"))
    pair = simulate_scatter_pa(_scatter_pa_request())
    for result in (mixed, fp32, pair):
        assert result["status"] == "predicted"
        json.dumps(result["prediction"])
        assert result["prediction"]["cost_record"]["stage_work"]
    stage = mixed["prediction"]["stage_work"][0]
    assert Fraction(stage["mte2"]) == 4100
    assert Fraction(stage["mte3"]) == 8192
    assert Fraction(stage["hbm"]) == 12292
    assert Fraction(mixed["prediction"]["cost_record"]["work"]["hbm"]) == 12292
    l2_stage = fp32["prediction"]["stage_work"][0]
    assert Fraction(l2_stage["mte2"]) == 8 * 64 * 2
    assert Fraction(l2_stage["mte3"]) == 8 * 64 * 4
    pair_stages = pair["prediction"]["stage_work"]
    assert len(pair_stages) == 2
    assert Fraction(pair_stages[0]["mte2"]) == 135168
    assert Fraction(pair_stages[0]["mte3"]) == 131072
    assert Fraction(pair_stages[1]["mte2"]) == 8192
    assert Fraction(pair_stages[1]["mte3"]) == 4096
    assert sum(Fraction(stage["mte2"]) for stage in pair_stages) == 143360
    assert sum(Fraction(stage["mte3"]) for stage in pair_stages) == 135168


def test_l2norm_column_and_row_tail_masks():
    column_tail = simulate_l2norm(_l2norm_rows_request(rows=8, head_dim=65))
    assert column_tail["status"] == "predicted"
    access = column_tail["program"]["access"]
    assert access["N"] == 65 and access["BD"] == 128
    assert access["issued_read_bytes"] == 8 * 65 * 2
    assert access["issued_write_bytes"] == 8 * 65 * 2
    assert access["compute_math_ops"] == 32 * 128 * 5
    assert column_tail["program"]["mask"]["valid_columns"] == 65
    assert column_tail["program"]["mask"]["padded_columns"] == 63
    assert column_tail["program"]["transaction_basis"]["source"] == "masked_M_N"
    row_tail = simulate_l2norm(_l2norm_rows_request(rows=33, head_dim=64))
    access = row_tail["program"]["access"]
    assert access["M"] == 33 and access["grid"] == 2
    assert access["tile_rows"] == 64 and access["tail_rows"] == 31
    assert access["issued_read_bytes"] == 33 * 64 * 2
    assert access["compute_math_ops"] == 64 * 64 * 5


def test_model_dynamic_gdn_partition_preserves_packed_requests():
    from types import SimpleNamespace
    from mindie_sim.prediction.piecewise.geometry import (
        RequestGeometry, WorkloadGeometry)
    from mindie_sim.prediction.piecewise.native_gdn import dynamic_gdn
    theta = SimpleNamespace(
        heads={"q": 8, "kv": 8, "dn": 128, "dv": 128},
        extras=(("chunk", 64),))
    operator = SimpleNamespace(name="gated_delta", theta=theta)
    compiled = SimpleNamespace(siblings=(operator,), operator=operator,
                               feature_metadata=())
    binding = SimpleNamespace(
        call_id="gdn-partition-test",
        api={"aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd"},
        typed={"attrs": {"chunk_size": 64}, "dtype": "bf16",
               "layout": "BSND", "signature_hash": "sig", "shapes": {}},
        source={"sha256": "5e3b0fb0", "revision": "r31"}, unresolved=())

    def prepare(lengths):
        geometry = WorkloadGeometry(
            tuple(RequestGeometry(length, 0, phase="prefill")
                  for length in lengths), phase="prefill")
        return dynamic_gdn(
            binding, compiled, geometry,
            validate=lambda _binding, actual: actual)

    even = prepare((64, 64))
    ragged = prepare((1, 127))
    assert even is not None and ragged is not None
    assert even["prepare"]["partition"]["cu_seqlens"] == [0, 64, 128]
    assert even["prepare"]["partition"]["chunks"] == 2
    assert even["prepare"]["partition"]["chunk_head_tasks"] == 16
    assert ragged["prepare"]["partition"]["cu_seqlens"] == [0, 1, 128]
    assert ragged["prepare"]["partition"]["chunks"] == 3
    assert ragged["prepare"]["partition"]["chunk_head_tasks"] == 24
    assert even["prepare"]["partition"]["state_rows"] == 2
    assert even["actual"]["shapes"]["q"] == [1, 128, 8, 128]
    assert ragged["actual"]["shapes"]["q"] == [1, 128, 8, 128]


def test_gdn_independent_matches_the_selected_fstate_call_report():
    from mindie_sim.prediction import call_gdn
    from mindie_sim.prediction.call import call_costs
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    from mindie_sim.schema.case import Case, Software
    from mindie_sim.schema.hardware import Hardware
    from mindie_sim.schema.model import ModelSpec, WorkloadSpec
    import importlib

    result = simulate_independent(_gdn_request(lengths=(64, 64)))
    assert result["status"] == "predicted"
    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    case = Case(
        machine=machine,
        model=ModelSpec(name="abi", layers=1, hidden_size=1, num_heads=1,
                        head_dim=1, intermediate_size=1, vocab_size=1),
        workload=WorkloadSpec._pure_model_sentinel(), runtime=Software())
    arguments = dict(result["program"]["call_program"]["arguments"])
    call = call_gdn.make_call("scan_prefill", {
        "batch": 2, "tokens": 128, "lengths": [64, 64],
        "v_heads": 8, "k_heads": 8, "k_dim": 128, "v_dim": 128,
        "chunk": 64, "chunk_size": 64, "phase": "prefill",
        "scale": 128 ** -0.5, "a_log": None, "dt_bias": None,
        "input_dtype": "bf16", "gate_dtype": "fp32", "beta_dtype": "bf16",
        "state_dtype": "fp32", "layout": "BSND",
        "use_exp2": True, "use_qk_l2norm_in_kernel": True,
        "allow_neg_eigval": False, "disable_recompute": True,
        "state_v_first": True, "output_final_state": True,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "return_intermediate_states": False, "initial_state": True,
    })
    report = call_costs(
        case, call.ops, recipes=call.recipes, mode="eager", trace_limit=128)
    assert result["program"]["call_report"]["operator_occurrences"] == (
        report["operator_occurrences"])
    assert result["program"]["call_report"]["execution"]["recipes"] == (
        report["execution"]["recipes"])
    assert float(Fraction(result["prediction"]["duration_ms"])) == pytest.approx(
        float(Fraction(str(report["duration_ms"]))), rel=1e-9, abs=1e-15)
    assert list(arguments["cu_seqlens"]) == [0, 64, 128]
    assert arguments["chunks"] == 2
    counts = result["program"]["native_kernel_counts"]
    assert counts == {"Cast": 1, "Transpose": 3, "Prepare32": 1,
                      "H24": 1, "O32": 1}
    outputs = result["program"]["call_program"]["outputs"]
    assert "final_state" in outputs
    state = result["program"]["state_value"]
    assert state["initial"]["shape"] == [2, 8, 128, 128]
    assert state["final"]["shape"] == [2, 8, 128, 128]
    assert result["program"]["state_records"]
    ports = result["program"]["state_ports"]
    assert len(ports) == 2
    assert [item["request_slot"] for item in ports] == [0, 1]
    assert [item["read_version"] for item in ports] == [0, 0]
    assert [item["write_version"] for item in ports] == [64, 64]
    assert all(item["producer"] == "chunk_gated_delta_rule_fwd" for item in ports)
    assert all(item["consumer"] == "scatter_ssm_states" for item in ports)
    assert all(item["last_use"] == "call_return_then_scatter" for item in ports)


def test_gdn_conditional_final_state_is_an_explicit_fstate_gap():
    from types import SimpleNamespace
    from mindie_sim.prediction.piecewise.geometry import (
        RequestGeometry, WorkloadGeometry)
    from mindie_sim.prediction.piecewise.native_gdn import (
        dynamic_gdn, simulate_independent)
    theta = SimpleNamespace(
        heads={"q": 8, "kv": 8, "dn": 128, "dv": 128},
        extras=(("chunk", 64),))
    operator = SimpleNamespace(name="gated_delta", theta=theta)
    compiled = SimpleNamespace(siblings=(operator,), operator=operator,
                               feature_metadata=())
    binding = SimpleNamespace(
        call_id="gdn-final-test",
        api={"aclnn_symbol": "aclnnChunkGatedDeltaRuleFwd"},
        typed={"attrs": {"chunk_size": 64, "output_final_state": False},
               "dtype": "bf16", "layout": "BSND",
               "signature_hash": "sig", "shapes": {}},
        source={"sha256": "5e3b0fb0", "revision": "r31"}, unresolved=())
    geometry = WorkloadGeometry(
        (RequestGeometry(128, 0, phase="prefill"),), phase="prefill")
    prepared = dynamic_gdn(
        binding, compiled, geometry,
        validate=lambda _binding, actual: actual)
    # F-STATE.call_gdn currently declares output_final_state=true for this
    # selected call; the model path truthfully stays unbound rather than
    # fabricating a conditional output.
    assert prepared is None
    request = _gdn_request(lengths=(64, 64))
    request["attrs"]["output_final_state"] = False
    result = simulate_independent(request)
    assert result["status"] == "no_match"
    assert any("output_final_state" in reason or "argument_not_selected"
               in reason for reason in result["reasons"])


def test_declared_dispatch_helper_reaches_leaf_and_parent_gap():
    from mindie_sim.prediction.piecewise.native_gdn import simulate_declared_abi
    import pytest
    assert simulate_declared_abi(_cat_request())["status"] == "predicted"
    parent = simulate_declared_abi(_scatter_pa_parent_request())
    assert parent["status"] == "not_implemented"
    with pytest.raises(Exception, match="leaf_abi_entry_not_declared"):
        simulate_declared_abi({"api": {"namespace": "torch", "entry": "unknown"},
                               "version": {}})


def test_rope_cos_sin_stride_is_consumed():
    request = _rope_request(tokens=4, heads=1, dim=128, rope_dim=64,
                            role="query")
    request["attrs"]["cos_stride0"] = 80
    request["attrs"]["sin_stride0"] = 96
    result = simulate_rope(request)
    assert result["status"] == "predicted"
    access = result["program"]["access"]
    assert access["cos_row_stride_bytes"] == 80 * 2
    assert access["sin_row_stride_bytes"] == 96 * 2
    assert result["occurrence"]["typed"]["attrs"]["cos_stride0"] == 80


def test_gdn_accepts_qsl_only_and_explicit_dense_physical_view():
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    packed = simulate_independent(_gdn_request(lengths=(64, 64)))
    qsl_only = simulate_independent(_gdn_request(
        lengths=(64, 64), include_lengths=False, qsl=[0, 64, 128]))
    dense = simulate_independent(_gdn_request(
        lengths=(64, 64), physical_batch=2,
        physical_axis_tokens=64, qsl=[0, 64, 128]))
    dense_qsl = simulate_independent(_gdn_request(
        lengths=(64, 64), physical_batch=2,
        physical_axis_tokens=64, include_lengths=False,
        qsl=[0, 64, 128]))
    for result in (qsl_only, dense, dense_qsl):
        assert result["status"] == "predicted"
        partition = result["program"]["partition"]
        assert partition["lengths"] == [64, 64]
        assert partition["cu_seqlens"] == [0, 64, 128]
        assert partition["chunks"] == 2
        assert partition["chunk_head_tasks"] == 16
        assert partition["state_rows"] == 2
        assert result["occurrence"]["typed"]["attrs"]["batch"] == 2
    assert dense["program"]["partition"]["physical_layout"] == "dense"
    assert dense["program"]["partition"]["view_transform"] == (
        "explicit_contiguous_BxT_to_1xBT")
    assert dense["program"]["partition"]["physical_batch"] == 2
    assert dense["program"]["partition"]["physical_axis_tokens"] == 64
    assert packed["program"]["partition"]["physical_layout"] == "packed"
    assert qsl_only["program"]["partition"]["physical_layout"] == "packed"
    assert qsl_only["prediction"]["cost_record"]["work"] == (
        packed["prediction"]["cost_record"]["work"])
    wrong = simulate_independent(_gdn_request(
        lengths=(64, 64), physical_batch=2,
        physical_axis_tokens=128, qsl=[0, 64, 128]))
    assert wrong["status"] == "no_match"
    assert any("dense_physical_batch_and_T_do_not_match_partition" in reason
               for reason in wrong["reasons"])


def test_gdn_complete_reduction_is_trace_independent_and_preserves_supply():
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    from mindie_sim.prediction.piecewise.compiler import PipeSupply
    from mindie_sim.schema.hardware import Hardware
    import importlib

    request = _gdn_request(lengths=(64, 64))
    off = simulate_independent(request, trace_limit=0)
    short = simulate_independent(request, trace_limit=128)
    full = simulate_independent(request, trace_limit=100000)
    assert off["status"] == short["status"] == full["status"] == "predicted"
    reference = off["prediction"]["cost_record"]
    for result in (short, full):
        cost = result["prediction"]["cost_record"]
        # Same structural program and service; trace retention must not
        # change the priced work or the completion clock.
        _assert_close(cost["work"], reference["work"])
        _assert_close(cost["stage_work"], reference["stage_work"])
        _assert_close(cost["duration_ms"], reference["duration_ms"])
        _assert_close(cost["materialization"], reference["materialization"])
        assert result["program"]["call_report"]["memory"] == (
            off["program"]["call_report"]["memory"])
        assert result["program"]["call_program"]["outputs"] == (
            off["program"]["call_program"]["outputs"])
        assert result["program"]["state_ports"] == off["program"]["state_ports"]
        assert result["program"]["state_records"] == off["program"]["state_records"]
    assert off["program"]["call_report"]["trace"]["retained"] == 0
    assert short["program"]["call_report"]["trace"]["retained"] == 128
    assert short["program"]["call_report"]["trace"]["dropped"] > 1700
    full_retained = full["program"]["call_report"]["trace"]["retained"]
    assert full_retained >= 128 + short["program"]["call_report"]["trace"]["dropped"]
    assert full["program"]["call_report"]["trace"]["dropped"] == 0
    stage = reference["stage_work"][0]
    resources = {row["resource"]: row for row in
                 off["program"]["call_report"]["resources"]}
    for resource, serviced in stage.items():
        assert Fraction(serviced) == Fraction(resources[resource]["work"])
    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    supply = PipeSupply.from_hardware(machine, dtype="bf16")
    explicit = simulate_independent(request, hardware=machine, supply=supply)
    assert explicit["status"] == "predicted"
    assert explicit["prediction"]["cost_record"]["work"] == reference["work"]
    assert explicit["prediction"]["cost_record"]["stage_work"] == (
        reference["stage_work"])
    assert explicit["program"]["partition"] == off["program"]["partition"]


def test_gdn_embedded_execution_submits_real_stages_without_new_context():
    from types import SimpleNamespace
    from mindie_sim.prediction import call_gdn
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.hardware import Hardware
    import importlib

    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    call = call_gdn.make_call("scan_prefill", {
        "batch": 2, "tokens": 128, "lengths": [64, 64],
        "cu_seqlens": [0, 64, 128], "v_heads": 8, "k_heads": 8,
        "k_dim": 128, "v_dim": 128, "chunk": 64, "chunk_size": 64,
        "phase": "prefill", "scale": 128 ** -0.5, "a_log": None,
        "dt_bias": None, "input_dtype": "bf16", "gate_dtype": "fp32",
        "beta_dtype": "bf16", "state_dtype": "fp32", "layout": "BSND",
        "use_exp2": True, "use_qk_l2norm_in_kernel": True,
        "allow_neg_eigval": False, "disable_recompute": True,
        "state_v_first": True, "output_final_state": True,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "return_intermediate_states": False, "initial_state": True,
    })
    context = SimpleNamespace(machine=machine, runtime=Software())
    isolated = simulate_independent(_gdn_request(), trace_limit=0)
    execution = Execution(
        context, CostModel(context), recipes=call.recipes, trace_limit=0,
        include_submission=False)
    shared = simulate_independent(
        _gdn_request(), hardware=machine, execution=execution)
    assert shared["status"] == "predicted"
    assert shared["execution"]["own_execution"] is False
    assert shared["execution"]["status"] == "pending"
    assert shared["program"]["call_report"]["own_execution"] is False
    assert shared["prediction"]["cost_record"]["stage_work"] == (
        isolated["prediction"]["cost_record"]["stage_work"])
    stage = shared["prediction"]["cost_record"]["stage_work"][0]
    assert shared["execution"]["work"] == isolated["execution"]["work"]
    # Embedded submission never advances the caller's engine internally.
    assert execution.engine.completed_count == 0
    handle = shared["execution_handle"]
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.done
    assert Fraction(str(execution.engine.serviced["memory@r0:s0"])) == Fraction(
        stage["memory@r0:s0"])
    # Compact execution (trace disabled) coalesces symmetric stage cohorts,
    # so the task count is not the detailed-decomposition count.  It must
    # still be far above the 18 operators, i.e. real per-stage work.
    assert shared["program"]["call_report"]["execution"][
        "completed_tasks"] > 300
    completed = execution.engine.completed_count
    assert completed > 0
    second = simulate_independent(
        _gdn_request(lengths=(1, 127)), hardware=machine,
        execution=execution)
    assert second["status"] == "predicted"
    assert second["execution"]["status"] == "pending"
    assert execution.engine.completed_count == completed
    second_handle = second["execution_handle"]
    execution.engine.advance(until=(second_handle.event,), first=False)
    assert execution.engine.completed_count > completed
    assert (second["program"]["call_report"]["execution"]["completed_tasks"]
            > shared["program"]["call_report"]["execution"]["completed_tasks"])
    assert second["program"]["partition"]["chunks"] == 3


def test_two_gdn_calls_share_one_engine_and_one_hbm_reduction():
    from types import SimpleNamespace
    from mindie_sim.prediction import call_gdn
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.hardware import Hardware
    import importlib

    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    context = SimpleNamespace(machine=machine, runtime=Software())

    def make(lengths):
        return call_gdn.make_call("scan_prefill", {
            "batch": len(lengths), "tokens": sum(lengths),
            "lengths": list(lengths), "cu_seqlens": [0, *(
                sum(lengths[:index]) for index in range(1, len(lengths) + 1))],
            "v_heads": 8, "k_heads": 8, "k_dim": 128, "v_dim": 128,
            "chunk": 64, "chunk_size": 64, "phase": "prefill",
            "scale": 128 ** -0.5, "a_log": None, "dt_bias": None,
            "input_dtype": "bf16", "gate_dtype": "fp32",
            "beta_dtype": "bf16", "state_dtype": "fp32", "layout": "BSND",
            "use_exp2": True, "use_qk_l2norm_in_kernel": True,
            "allow_neg_eigval": False, "disable_recompute": True,
            "state_v_first": True, "output_final_state": True,
            "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
            "return_intermediate_states": False, "initial_state": True,
        })

    first, second = make((64, 64)), make((1, 127))
    execution = Execution(
        context, CostModel(context), recipes=first.recipes, trace_limit=0,
        include_submission=False)
    steps = []
    for call in (first, second):
        iteration = IterationWork(
            call.ops, IterationPlan(()), 1, 0, 0, None, "graph",
            "declared_graph", replay_scope="none")
        steps.append(execution.add(iteration, 0, 0))
    execution.engine.advance(
        until=tuple(step.completion for step in steps), first=False)
    shared_time = execution.engine.now
    shared_completed = execution.engine.completed_count
    assert steps[0].completion.time <= steps[1].completion.time
    memory = execution.engine.serviced["memory@r0:s0"]
    isolated, isolated_times, isolated_completed = [], [], []
    for call in (first, second):
        solo = Execution(
            context, CostModel(context), recipes=call.recipes,
            trace_limit=0, include_submission=False)
        iteration = IterationWork(
            call.ops, IterationPlan(()), 1, 0, 0, None, "graph",
            "declared_graph", replay_scope="none")
        step = solo.add(iteration, 0, 0)
        solo.engine.advance(until=(step.completion,), first=False)
        isolated.append(solo.engine.serviced["memory@r0:s0"])
        isolated_times.append(solo.engine.now)
        isolated_completed.append(solo.engine.completed_count)
    assert abs(memory - sum(isolated)) < 1e-6
    assert sum(isolated_completed) <= shared_completed <= sum(isolated_completed) + 8
    assert shared_time > max(isolated_times)


def test_rope_embedded_execution_uses_caller_context():
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope_entry)
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.hardware import Hardware
    import importlib

    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    context = SimpleNamespace(machine=machine, runtime=Software())
    execution = Execution(
        context, CostModel(context), trace_limit=0,
        include_submission=False)
    result = simulate_rope_entry(
        _rope_request(tokens=32, heads=1, dim=128, rope_dim=64,
                      role="query"),
        hardware=machine, execution=execution)
    assert result["status"] == "predicted"
    assert result["execution"]["own_execution"] is False
    assert result["execution"]["status"] == "pending"
    assert execution.engine.completed_count == 0
    handle = result["execution_handle"]
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.done
    assert execution.engine.completed_count > 0
    assert Fraction(result["execution"]["work"]["hbm"]) == 12288
    assert result["execution_handle"].to_dict()["terminal"]["done"] is True


def test_gdn_and_rope_reuse_one_execution_and_preserve_work():
    from types import SimpleNamespace
    from mindie_sim.prediction import call_gdn
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.native_gdn import simulate_independent
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope)
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.hardware import Hardware
    import importlib

    machine = Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))
    call = call_gdn.make_call("scan_prefill", {
        "batch": 2, "tokens": 128, "lengths": [64, 64],
        "cu_seqlens": [0, 64, 128], "v_heads": 8, "k_heads": 8,
        "k_dim": 128, "v_dim": 128, "chunk": 64, "chunk_size": 64,
        "phase": "prefill", "scale": 128 ** -0.5, "a_log": None,
        "dt_bias": None, "input_dtype": "bf16", "gate_dtype": "fp32",
        "beta_dtype": "bf16", "state_dtype": "fp32", "layout": "BSND",
        "use_exp2": True, "use_qk_l2norm_in_kernel": True,
        "allow_neg_eigval": False, "disable_recompute": True,
        "state_v_first": True, "output_final_state": True,
        "use_gate_in_kernel": False, "use_beta_sigmoid_in_kernel": False,
        "return_intermediate_states": False, "initial_state": True,
    })
    context = SimpleNamespace(machine=machine, runtime=Software())
    execution = Execution(
        context, CostModel(context), recipes=call.recipes, trace_limit=0,
        include_submission=False)
    gdn_alone = simulate_independent(_gdn_request(), trace_limit=0)
    rope_request = _rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query")
    rope_alone = simulate_rope(rope_request, trace_limit=0)
    gdn = simulate_independent(
        _gdn_request(), hardware=machine, execution=execution)
    rope = simulate_rope(
        rope_request, hardware=machine, execution=execution)
    assert gdn["execution"]["own_execution"] is False
    assert rope["execution"]["own_execution"] is False
    assert execution.engine.completed_count == 0
    gdn_handle = gdn["execution_handle"]
    rope_handle = rope["execution_handle"]
    execution.engine.advance(
        until=(gdn_handle.event, rope_handle.event), first=False)
    assert gdn_handle.done and rope_handle.done
    assert gdn["prediction"]["cost_record"]["work"] == (
        gdn_alone["prediction"]["cost_record"]["work"])
    assert rope["prediction"]["cost_record"]["work"] == (
        rope_alone["prediction"]["cost_record"]["work"])
    gdn_hbm = Fraction(gdn["execution"]["work"]["hbm_bytes"])
    rope_hbm = Fraction(rope["execution"]["work"]["hbm"])
    memory = Fraction(str(execution.engine.serviced["memory@r0:s0"]))
    assert abs(memory - (gdn_hbm + rope_hbm)) < Fraction(1, 1000)


def _a5_machine():
    import importlib
    from mindie_sim.schema.hardware import Hardware
    return Hardware.from_mapping(
        importlib.import_module("mindie_sim.backends.ascend").materialize(
            {"sku": "Ascend950DT_9582"}))


def _idle_execution(machine, *, trace_limit=0):
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.schema.case import Software
    context = SimpleNamespace(machine=machine, runtime=Software())
    return Execution(context, CostModel(context), trace_limit=trace_limit,
                     include_submission=False)


def test_mixed_scatter_submits_same_program_once_to_idle_execution():
    from mindie_sim.prediction.piecewise.native_scatter_ssm_states import (
        simulate_independent as simulate_scatter)
    machine = _a5_machine()
    execution = _idle_execution(machine)
    result = simulate_scatter(_scatter_request(
        batch=1, heads=1, dk=64, dv=32, state_dtype="fp32",
        source_dtype="fp16", branch="use_fused_chunk"),
        hardware=machine, execution=execution)
    assert result["status"] == "predicted"
    assert result["execution"]["own_execution"] is False
    assert result["execution"]["status"] == "pending"
    assert execution.engine.completed_count == 0
    assert len(result["occurrences"]) == 1
    assert result["occurrences"][0]["charge"] == "call"
    assert Fraction(result["execution"]["work"]["mte2"]) == 4100
    assert Fraction(result["execution"]["work"]["mte3"]) == 8192
    assert Fraction(result["execution"]["work"]["hbm"]) == 12292
    assert result["program"]["state_handles"]["writers"][0]["tensor"] == "state"
    assert result["execution_program"].call is None
    # exactly this occurrence is submitted once, and the terminal is real
    handle = result["execution_handle"]
    assert handle is result["execution_handle"]
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.done
    assert execution.engine.completed_count >= 1
    assert Fraction(str(execution.engine.serviced["mte2_aiv@r0:s0"])) == 4100
    assert Fraction(str(execution.engine.serviced["mte3@r0:s0"])) == 8192
    assert Fraction(str(execution.engine.serviced["memory@r0:s0"])) == 12292
    assert handle.to_dict()["terminal"]["done"] is True
    assert result["program"]["prepared_program"]["kind"] == "parts"
    assert result["program"]["prepared_program"]["parts"]
    assert "start_signature" in result["program"]["prepared_program"]


def test_explicit_supply_enters_standalone_execution_rates():
    from dataclasses import replace
    from mindie_sim.prediction.piecewise.compiler import PipeSupply
    from mindie_sim.prediction.piecewise.native_gdn import (
        simulate_independent as simulate_gdn_entry)
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope_entry)
    machine = _a5_machine()
    base = PipeSupply.from_hardware(machine, dtype="bf16")
    low = replace(base, rates=tuple(
        (name, 1.0 if name == "hbm" else value)
        for name, value in base.rates))
    rope = simulate_rope_entry(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"),
        hardware=machine, supply=low)
    assert rope["status"] == "predicted"
    assert rope["execution"]["own_execution"] is True
    assert Fraction(rope["execution"]["work"]["hbm"]) == 12288
    assert Fraction(rope["execution"]["duration_ms"]) >= 12288
    memory = next(row for row in rope["execution"]["resources"]
                  if row["resource"] == "memory@r0:s0")
    assert memory["capacity_per_ms"] == 1.0
    gdn = simulate_gdn_entry(_gdn_request(), hardware=machine, supply=low)
    assert gdn["status"] == "predicted"
    hbm_work = Fraction(gdn["execution"]["work"]["hbm_bytes"])
    assert Fraction(gdn["execution"]["duration_ms"]) >= hbm_work


def test_rope_after_copy_attributes_each_occurrence_and_waits_ready():
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope_entry)
    machine = _a5_machine()
    execution = _idle_execution(machine)
    copy = execution.engine.submit(
        "copy:4096", {"memory@r0:s0": 4096}, owner="native:copy")
    ready = execution.engine.submit(
        "input_ready:5ms", {}, release=5.0, owner="native:input")
    result = simulate_rope_entry(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"),
        hardware=machine, execution=execution, ready=ready)
    assert result["status"] == "predicted"
    assert result["execution"]["status"] == "pending"
    assert Fraction(result["execution"]["work"]["hbm"]) == 12288
    assert result["execution"]["engine_reconciliation"]["serviced"] == {}
    handle = result["execution_handle"]
    assert handle.start_event is ready
    execution.engine.advance(until=(copy, handle.event), first=False)
    assert copy.done and handle.done
    assert handle.event.time >= 5.0
    assert Fraction(str(execution.engine.serviced["memory@r0:s0"])) == 16384
    assert {name: Fraction(str(value))
            for name, value in handle.work} == {
        name: Fraction(value) for name, value in result["execution"]["work"].items()}
    assert Fraction(result["occurrences"][0]["priced"]["work"]["hbm"]) == 12288
    reconciled = handle.reconcile()
    assert Fraction(reconciled["serviced"]["memory@r0:s0"]) == 16384
    assert reconciled["used_for_occurrence_work"] is False


def test_non_default_owner_and_placement_rebind_is_recorded():
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope_entry)
    machine = _a5_machine()
    execution = _idle_execution(machine)
    result = simulate_rope_entry(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"),
        hardware=machine, execution=execution,
        owner="native:leaf:alt", replica=0, stage=0)
    assert result["status"] == "predicted"
    assert result["execution"]["owner"] == "native:leaf:alt"
    assert result["execution"]["placement"] == [0, 0]
    handle = result["execution_handle"]
    assert handle.owner == "native:leaf:alt"
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.owner == "native:leaf:alt"
    assert handle.done
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.schema.case import Software
    from types import SimpleNamespace
    from dataclasses import replace as dataclass_replace
    runtime = Software(parallel=ParallelSpec(dp=2))
    second = dataclass_replace(machine.dies[0], rank=1)
    two = dataclass_replace(machine, dies=(machine.dies[0], second))
    context = SimpleNamespace(machine=two, runtime=runtime)
    placed = Execution(context, CostModel(context), trace_limit=0,
                       include_submission=False)
    moved = simulate_rope_entry(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"),
        hardware=machine, execution=placed, owner="native:leaf:r1",
        replica=1, stage=0)
    assert moved["execution"]["placement"] == [1, 0]
    placed_handle = moved["execution_handle"]
    assert "@1:s0" in placed_handle.event.name
    placed.engine.advance(until=(placed_handle.event,), first=False)
    assert placed_handle.done


def test_scatter_pa_children_submit_once_each_and_keep_two_occurrences():
    from mindie_sim.prediction.piecewise.native_r64_scatter_pa_cache import (
        simulate_independent as simulate_scatter_pa)
    machine = _a5_machine()
    execution = _idle_execution(machine)
    result = simulate_scatter_pa(_scatter_pa_request(), hardware=machine,
                                 execution=execution)
    assert result["status"] == "predicted"
    assert result["execution"]["status"] == "pending"
    assert execution.engine.completed_count == 0
    assert len(result["occurrences"]) == 2
    assert all(item["charge"] == "call"
               for item in result["occurrences"])
    assert sum(Fraction(item["priced"]["work"]["mte2"])
               for item in result["occurrences"]) == 143360
    assert sum(Fraction(item["priced"]["work"]["mte3"])
               for item in result["occurrences"]) == 135168
    handle = result["execution_handle"]
    assert len(handle.part_spans) == 2
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.done
    assert Fraction(str(execution.engine.serviced["mte2_aiv@r0:s0"])) == 143360
    assert Fraction(str(execution.engine.serviced["mte3@r0:s0"])) == 135168
    assert Fraction(result["execution"]["work"]["mte2"]) == 143360
    assert Fraction(result["execution"]["work"]["mte3"]) == 135168


def test_gdn_grant_releases_only_after_last_async_reader():
    from mindie_sim.prediction.piecewise.native_gdn import (
        simulate_independent as simulate_gdn_entry)
    machine = _a5_machine()
    execution = _idle_execution(machine)
    last_reader = execution.engine.submit(
        "downstream:last_read", {}, release=10.0,
        owner="native:downstream")
    pool = ("l1@r0:s0" if "l1@r0:s0" in execution.engine.pools
            else next(iter(execution.engine.pools)))
    result = simulate_gdn_entry(
        _gdn_request(), hardware=machine, execution=execution,
        output_grants={"final_state": {
            "pool": pool, "size": 4096, "last_reader": last_reader}},
        owner="native:gdn:alt")
    assert result["status"] == "predicted"
    handle = result["execution_handle"]
    assert handle.grants
    assert handle.release_events
    execution.engine.advance(
        until=(handle.event, last_reader, handle.release_events[0]),
        first=False)
    assert last_reader.done
    assert handle.release_events[0].done
    assert handle.release_events[0].time >= 10.0
    final = [item for item in handle.writers
             if item["tensor"] == "final_state"]
    assert final and final[0]["event"] is handle.event


def test_prepared_program_is_launchable_once_by_the_next_consumer():
    from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
        simulate_independent as simulate_rope_entry)
    machine = _a5_machine()
    prepared = simulate_rope_entry(_rope_request(
        tokens=32, heads=1, dim=128, rope_dim=64, role="query"),
        hardware=machine, prepare_only=True)
    assert prepared["status"] == "predicted"
    assert prepared["execution"] is None
    assert prepared["submission"] is None
    program = prepared["execution_program"]
    assert program is prepared["execution_program"]
    assert program.prepared is False
    execution = _idle_execution(machine)
    ready = execution.engine.submit("prepared:input_ready", {}, release=5.0)
    handle = program.start(execution, ready=ready)
    assert program.prepared is True
    assert execution.engine.completed_count == 0
    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.done
    assert handle.event.time >= 5.0
    assert {name: Fraction(str(value))
            for name, value in handle.work}["hbm"] == 12288
    # The same prepared object is not rebuilt on a second start.
    second = program.start(execution)
    assert second is not handle
    assert program.to_dict()["prepared"] is True
