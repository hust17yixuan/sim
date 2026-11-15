"""Pinned v4 exact transfer, packing, and lifetime vectors.

[Role]
Compare public transfer/planning APIs and arithmetic primitives with the v4
JSON values, plus independent small brute-force lattice/address oracles.
[Invariants]
Expected bytes are integers; no tolerances, rounded floats, or skipped vectors.
Oracle loops are test-only; production Load3D never enumerates windows.
[Timeouts & Locks]
Bounded deterministic tests with no waits or locks.
[Side-Effect Policy]
Read packaged fixtures only; no filesystem writes.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.opflow import BufferPool, BufferSpec, get_profile, plan_buffer_pool
from mindie_sim.opflow.exact_math import (
    MathDomainError, align, floor_sum, fractal_bit_offset, lattice_count,
    layout_capacity, load3d_counts, nddma_counts,
)
from mindie_sim.opflow.lifetime_math import allocation_peaks
from mindie_sim.opflow.transfer_accounting import GmStrides, TransferRequest, account_transfer

V = json.loads((Path(__file__).parent / "fixtures/v4_exact_vectors.json").read_text())["exact_test_tables"]
P = get_profile("Ascend950PR")


@pytest.mark.parametrize("vector", V["load3d_vectors"], ids=lambda v: v["id"])
def test_load3d_v4_public_and_exact(vector):
    p, expected = vector["parameters"], vector["expected"]
    assert load3d_counts(**p) == expected
    for policy, key in (("full_source_once", "actual_source_read_bytes_once"),
                        ("per_reference", "actual_source_read_bytes_per_ref")):
        result = account_transfer(TransferRequest(
            "load3d", source_hw=(p["H"], p["W"]), channels=p["C"],
            kernel=(p["kh"],p["kw"]), stride=(p["sh"],p["sw"]),
            dilation=(p["dh"],p["dw"]), pad_before=(p["pt"],p["pl"]),
            pad_after=(p["pb"],p["pr"]), bits_per_element=p["bits"],
            layout=p.get("layout","ND"), read_policy=policy,
            src_pool=BufferPool.L1, dst_pool=BufferPool.L0A), P)
        assert result.actual_read_bytes == expected[key]
        assert result.actual_write_bytes == expected["actual_dest_write_bytes"]
        assert result.logical_input_bytes == expected["logical_input_bytes"]
        assert result.raw_destination_capacity_bytes == expected["dest_physical_capacity_bytes"]
        assert result.destination_capacity_bytes == align(expected["dest_physical_capacity_bytes"],512)
        assert result.output_hw == (expected["output_h"],expected["output_w"])


def test_floor_sum_and_lattice_independent_oracles():
    for n in range(8):
        for m in range(1,8):
            for a in range(-6,7):
                for b in (-10,-1,0,1,9):
                    assert floor_sum(n,m,a,b) == sum((a*i+b)//m for i in range(n))
    for outputs in range(7):
        for kernel in range(1,6):
            for stride in (1,2,4):
                for dilation in (1,3):
                    for z in range(-2,22):
                        assert lattice_count(outputs,kernel,stride,dilation,z) == sum(
                            stride*i+dilation*j <= z for i in range(outputs) for j in range(kernel))
    # Huge extents would make window enumeration infeasible; the answer is exact.
    n = 10**30 + 17
    result = load3d_counts(H=n,W=n,C=2,kh=1,kw=1,bits=4)
    assert result["actual_source_read_bytes_per_ref"] == n*n


@pytest.mark.parametrize("vector", V["fractal_vectors"])
def test_fractal_capacity_and_address_map(vector):
    m,n,q,layout = (vector[x] for x in ("m","n","bits","layout"))
    assert layout_capacity(m,n,q,layout) == vector["capacity"]
    assert layout_capacity(m,n,q,"ND") == vector["logical_row_packed_bytes"]
    offsets = {fractal_bit_offset(m,n,q,i,j,layout) for i in range(m) for j in range(n)}
    assert len(offsets) == m*n
    assert max(offsets)+q == vector["max_live_bit"]
    result = account_transfer(TransferRequest("dn2nz", rows=m,cols=n,
                              bits_per_element=q,fractal_layout=layout,packing="row_restart",
                              dst_pool=BufferPool.L1),P)
    assert result.raw_destination_capacity_bytes == vector["capacity"]
    assert result.logical_input_bytes == vector["logical_row_packed_bytes"]


def test_orientation_accumulator_and_packed_weight_are_distinct():
    assert layout_capacity(17,9,32,"NZ") == 2048
    assert layout_capacity(17,9,32,"ZN") == 1536
    assert layout_capacity(17,17,32,"NZ") == 3072
    assert layout_capacity(16,16,32,"L0C") == 1024
    assert layout_capacity(32,16,4,"MXA8W4") == 256
    assert layout_capacity(16,64,4,"NZ") == 512


@pytest.mark.parametrize("vector", V["nddma_vectors"], ids=lambda v: v["id"])
def test_nddma_v4_and_public_payload(vector):
    args = {k:vector[k] for k in ("shape","stride_bits","element_bits","base_bits","alignment")}
    assert nddma_counts(**args) == vector["expected"]
    result = account_transfer(TransferRequest("nddma", nd_shape=tuple(args["shape"]),
                              stride_bits=tuple(args["stride_bits"]),bits_per_element=args["element_bits"],
                              base_bits=args["base_bits"]),P)
    assert result.actual_read_bytes == result.actual_write_bytes == vector["expected"]["actual_transfer_bytes"]
    assert result.address_span_bytes == vector["expected"]["address_span_bytes"]


def test_nddma_full_pitch_and_overlap_rejection():
    r = GmStrides(3,20,256)
    assert (r.effective_bytes,r.address_span_bytes,r.full_pitch_reserve) == (60,532,768)
    assert align(r.full_pitch_reserve,512) == 1024
    req = TransferRequest("gm_strided",rows=3,cols=20,element_bytes=1,gm_strides=r,capacity_policy="full_pitch")
    result = account_transfer(req,P,pool=BufferPool.GM)
    assert result.actual_read_bytes == 60 and result.source_capacity_bytes == 768
    with pytest.raises(MathDomainError):
        nddma_counts((2,2),(8,8),8)
    with pytest.raises(MathDomainError):
        load3d_counts(H=1,W=1,C=1,kh=1,kw=1,bits=4)


def test_interval_vector_and_physical_alias_ownership():
    v = V["interval_vector"]
    result = allocation_peaks(v["buffers"],{"UB":32,"L1":512},v["S_max"])
    assert {str(k):value for k,value in result["peaks"].items()} == v["expected"]
    a = BufferSpec("a",BufferPool.UB_LANE0,100,slots=3,first_stage=2,last_stage=4,alloc_id="backing")
    view = replace(a,name="view",bytes_per_tile=80,last_stage=7)
    other = replace(a,name="other",alloc_id="different")
    assert plan_buffer_pool((a,view),BufferPool.UB_LANE0,P).peak_bytes == 384
    assert plan_buffer_pool((a,view,other),BufferPool.UB_LANE0,P).peak_bytes == 768
    owners = plan_buffer_pool((a,replace(other,owner=1)),BufferPool.UB_LANE0,P)
    assert owners.peak_by_owner == {0:384,1:384} and owners.peak_bytes == 384
    adjacent = replace(other,first_stage=5,last_stage=8)
    assert plan_buffer_pool((adjacent,a),BufferPool.UB_LANE0,P).peak_bytes == 384
    overlap = replace(adjacent,first_stage=4)
    assert plan_buffer_pool((a,overlap),BufferPool.UB_LANE0,P).peak_bytes == 768
    anonymous = replace(a,alloc_id=None)
    assert plan_buffer_pool((anonymous,anonymous),BufferPool.UB_LANE0,P).peak_bytes == 768
