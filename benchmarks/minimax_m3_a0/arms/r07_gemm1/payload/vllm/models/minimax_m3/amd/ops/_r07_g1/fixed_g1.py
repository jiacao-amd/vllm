# SPDX-License-Identifier: Apache-2.0
"""Native G1 configuration already validated on the real-layer component."""

from functools import cache

KERNEL_NAMES = ("r07_gemm1_slot_a16w4_h6144_i768_ne129_bm16_tn64_swiglu_xcd1_w2_kw2",)


@cache
def compile_fixed():
    from .gemm1_slot import compile_gemm1_a16w4_port

    return compile_gemm1_a16w4_port(
        BM=16,
        D_HIDDEN=6144,
        D_INTER=768,
        NE=129,
        TOPK=5,
        TILE_N=64,
        TILE_K=128,
        waves_per_eu=2,
        k_wave=2,
        act="swiglu",
        b_cache_mod=2,
        xcd_swizzle=1,
        w_dtype="fp4",
        w_layout="standard",
        use_k16=False,
    )
