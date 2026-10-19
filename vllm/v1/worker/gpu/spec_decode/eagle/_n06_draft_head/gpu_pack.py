"""One-time raw uint16 packing outside decode, capture and formal timing."""

import triton as tr
import triton.language as tl


@tr.jit
def pack_bf16_words(
    source,
    destination,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ki = offset % BK
    ni = offset // BK % BN
    kt = offset // (BK * BN) % (K // BK)
    nt = offset // (BK * BN * (K // BK))
    n, k = nt * BN + ni, kt * BK + ki
    safe_n = tl.where(n < N, n, 0)
    word = tl.load(source + safe_n * K + k, offset < TOTAL, other=0)
    tl.store(destination + offset, word, offset < TOTAL)
