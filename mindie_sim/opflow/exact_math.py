"""Exact transfer, packing, lane, and memory-service arithmetic.

[Role]
Implement the common integer/rational algebra pinned by the v4 math pack.
[Invariants]
Payload, address span, raw layout size, and aligned allocation are distinct.
No dimension passes through float; bool is not a count. Unsupported bit-stride
layouts fail closed. Load3D uses a signed Euclidean floor sum, never windows.
[Timeouts & Locks]
Load3D is logarithmic in the stride/dilation coefficients; no I/O or locks.
[Side-Effect Policy]
Pure arithmetic; no filesystem, device, or environment changes.
"""

from fractions import Fraction
from math import prod


class MathDomainError(ValueError):
    """A mathematical domain or selected-layout contract is violated."""


def integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise MathDomainError(f"{name} must be an integer >= {minimum}")
    return value


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise MathDomainError("expected a finite real number")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except TimeoutError:
        raise
    except (ValueError, OverflowError) as exc:
        raise MathDomainError("expected a finite real number") from exc


def ceildiv(n, d):
    integer("numerator", n)
    integer("denominator", d, 1)
    return (n + d - 1) // d


def align(n, a):
    return ceildiv(n, a) * a


def floor_sum(n, m, a, b):
    """Signed floor sum: O(log m) Euclidean iterations, constant small-n exit.

    With the fixed n<=7 termination this also gives O(log(min(n,m)+1))
    iterations (unit-cost integer arithmetic, not a bit-complexity claim).
    Reduced a,b satisfy n'<=n, sometimes equality. Across two swaps, with
    r=m%a<m/2, n'' < (r/m)*n + r/a + 1 < n/2+2. Thus n>=8 shrinks by
    at least a factor 3/4 every two swaps; smaller n takes at most seven terms.
    Neither an input-sized scan nor a modulus-sized scan is performed.
    """
    integer("n", n)
    integer("m", m, 1)
    if type(a) is not int or type(b) is not int:
        raise MathDomainError("floor-sum coefficients must be integers")
    answer = 0
    while True:
        if n == 0:
            return answer
        if n == 1:
            return answer + b // m
        if a == 0:
            return answer + n * (b // m)
        if n <= 7:
            return answer + sum((a * i + b) // m for i in range(n))
        qa, a = divmod(a, m)
        qb, b = divmod(b, m)
        answer += qa * n * (n - 1) // 2 + qb * n
        if a == 0:
            return answer
        y = a * n + b
        if y < m:
            return answer
        n, b = divmod(y, m)
        m, a = a, m


def lattice_count(outputs, kernel, stride, dilation, z):
    integer("outputs", outputs)
    integer("kernel", kernel, 1)
    integer("stride", stride, 1)
    integer("dilation", dilation, 1)
    if type(z) is not int:
        raise MathDomainError("lattice bound must be an integer")
    if outputs == 0:
        return 0
    a = min(kernel, max(0, 1 + (z - stride * (outputs - 1)) // dilation))
    b = min(kernel, max(0, 1 + z // dilation))
    return a * outputs + b - a + floor_sum(b - a, stride, -dilation, z - dilation * a)


def out_axis(size, kernel, stride, dilation, before, after):
    for name, value, low in (("size", size, 0), ("kernel", kernel, 1),
                             ("stride", stride, 1), ("dilation", dilation, 1),
                             ("before", before, 0), ("after", after, 0)):
        integer(name, value, low)
    return max(0, 1 + (size + before + after - dilation * (kernel - 1) - 1) // stride)


def valid_axis(size, kernel, stride, dilation, before, after):
    outputs = out_axis(size, kernel, stride, dilation, before, after)
    return (lattice_count(outputs, kernel, stride, dilation, size - 1 + before)
            - lattice_count(outputs, kernel, stride, dilation, before - 1))


def layout_capacity(m, n, bits, layout="NZ"):
    integer("m", m)
    integer("n", n)
    integer("bits", bits, 1)
    if layout == "ND":
        return m * ceildiv(n * bits, 8)
    if layout in ("NZ", "ZN") and bits in (4, 8, 16, 32):
        c0 = 256 // bits
        a, b = (16, c0) if layout == "NZ" else (c0, 16)
    elif layout == "L0C" and bits == 32:
        a, b = 16, 16
    elif layout == "MXA8W4" and bits == 4:
        a, b = 32, 16  # axes K,N; this is a 256B packed-weight block
    else:
        raise MathDomainError("unsupported layout/element-bit combination")
    return align(m, a) * align(n, b) * bits // 8


def fractal_bit_offset(m, n, bits, i, j, layout="NZ"):
    layout_capacity(m, n, bits, layout)
    integer("i", i)
    integer("j", j)
    if i >= m or j >= n or layout not in ("NZ", "ZN"):
        raise MathDomainError("fractal coordinate outside logical extent")
    if layout == "ZN":
        m, n, i, j = n, m, j, i
    c0 = 256 // bits
    return bits * (((j // c0) * ceildiv(m, 16) + i // 16) * 16 * c0
                   + (i % 16) * c0 + j % c0)


def load3d_counts(*, H, W, C, kh, kw, sh=1, sw=1, dh=1, dw=1,
                  pt=0, pb=0, pl=0, pr=0, bits=16, layout="ND", alignment=1):
    integer("C", C, 1)
    integer("bits", bits, 1)
    integer("alignment", alignment, 1)
    if C * bits % 8:
        raise MathDomainError("sub-byte channels require an explicit bit-address map")
    oh, ow = out_axis(H, kh, sh, dh, pt, pb), out_axis(W, kw, sw, dw, pl, pr)
    rh, rw = valid_axis(H, kh, sh, dh, pt, pb), valid_axis(W, kw, sw, dw, pl, pr)
    source = ceildiv(H * W * C * bits, 8)
    output = oh * ow * ceildiv(kh * kw * C * bits, 8)
    capacity = layout_capacity(oh * ow, kh * kw * C, bits, layout)
    return dict(output_h=oh, output_w=ow, valid_references=C * rh * rw,
                logical_input_bytes=source,
                actual_source_read_bytes_once=source if oh * ow else 0,
                actual_source_read_bytes_per_ref=rh * rw * C * bits // 8,
                unfolded_logical_output_bytes=output, actual_dest_write_bytes=output,
                dest_physical_capacity_bytes=capacity,
                pool_allocation_bytes=align(capacity, alignment))


def nddma_counts(shape, stride_bits, element_bits, base_bits=0, alignment=1):
    """Exact disjoint rectangular bit layouts with byte-aligned outer strides.

    Collapse contiguous inner axes into one bit interval. Outer axes must have
    non-overlapping byte spans, so their byte unions can be multiplied exactly.
    Nonhierarchical/negative/overlapping layouts require an explicit address map.
    """
    integer("element_bits", element_bits, 1)
    integer("base_bits", base_bits)
    integer("alignment", alignment, 1)
    if not shape or len(shape) != len(stride_bits):
        raise MathDomainError("shape and strides must have equal nonzero rank")
    for n, s in zip(shape, stride_bits):
        integer("extent", n)
        integer("stride_bits", s)
    if 0 in shape:
        return dict(logical_bits=0, actual_transfer_bytes=0, address_span_bytes=0,
                    physical_capacity_bytes=0)
    width, blocks, span_bits, outer = element_bits, 1, element_bits, False
    for s, n in sorted(zip(stride_bits, shape)):
        if n == 1:
            continue
        if s < span_bits:
            raise MathDomainError("overlapping or nonhierarchical element strides")
        if not outer and s == width:
            width *= n
        else:
            outer = True
            if s % 8 or s < align(span_bits + base_bits % 8, 8):
                raise MathDomainError("outer bit-stride byte union needs an explicit address map")
            blocks *= n
        span_bits += (n - 1) * s
    payload = blocks * ceildiv(base_bits % 8 + width, 8)
    span = ceildiv(base_bits % 8 + span_bits, 8)
    return dict(logical_bits=prod(shape) * element_bits, actual_transfer_bytes=payload,
                address_span_bytes=span, physical_capacity_bytes=align(span, alignment))


def memory_seconds(read, write, read_bps, write_bps, alpha=Fraction(3, 10)):
    read, write, rb, wb, a = map(rational, (read, write, read_bps, write_bps, alpha))
    if min(read, write) < 0 or min(rb, wb) <= 0 or not 0 <= a <= 1:
        raise MathDomainError("invalid memory service domain")
    tr, tw = read / rb, write / wb
    return max(tr, tw) + a * min(tr, tw)


def lane_service(work, rates):
    if len(work) != 2 or len(rates) != 2:
        raise MathDomainError("one C pair requires two lane records")
    w0, w1, r0, r1 = map(rational, (*work, *rates))
    if min(w0, w1) < 0 or min(r0, r1) <= 0:
        raise MathDomainError("invalid lane work/rate")
    return max(w0 / r0, w1 / r1)


def one_c_two_v(cube, lanes):
    """Sum explicit C, lane-0, and lane-1 counters once, including broadcast."""
    if len(lanes) != 2:
        raise MathDomainError("one C pair requires exactly two lane records")
    keys = ("work", "read", "write")
    for record in (cube, *lanes):
        if set(record) != set(keys):
            raise MathDomainError("each engine needs work/read/write counters")
        for key in keys:
            integer(key, record[key])
    return {key: cube[key] + sum(lane[key] for lane in lanes) for key in keys}


def semantic_vector_costs(dim):
    """Unsplit scalar operation conventions; FMA=2, conversions separate."""
    d = integer("dim", dim, 1)
    return {"D": d, "attnres": 12*d+15, "engram": 9*d+16,
            "add_rms": 5*d+3, "replay_unsplit": 7*d*d+15*d+8, "commit": 3*d*d}


def replay_u_promotion(heads, accepted, dim):
    n = integer("heads", heads, 1) * integer("accepted", accepted) * integer("dim", dim, 1)
    return {"read": 2 * n, "write": 4 * n, "casts": n}


def cache_service(record_bytes, references, cold_fills=1):
    integer("record_bytes", record_bytes)
    integer("references", references)
    integer("cold_fills", cold_fills)
    if cold_fills > references:
        raise MathDomainError("cold fills exceed references")
    return {"unique_working_set": record_bytes if references else 0,
            "l2_read_service": record_bytes * references,
            "hbm_read_service": record_bytes * cold_fills}
