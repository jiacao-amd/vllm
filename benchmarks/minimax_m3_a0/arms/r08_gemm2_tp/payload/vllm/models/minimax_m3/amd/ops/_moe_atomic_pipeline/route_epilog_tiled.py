# SPDX-License-Identifier: Apache-2.0
# Adapted from A35's installed-AITER A16W4 BF16 route epilogue.
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Write identical BF16 contributions in [column, token, slot, inner] order."""
from types import FunctionType
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import gemm2
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load

@flyc.jit
def _route_tiled_epilog(lds_acc_base_i32, accm, arg_out, arg_stids, arg_sweights, m_row, n_block_idx, wave, lane, i32_M, BM, N_OUT, BN):
    """Change only the route address, not the multiply or BF16 conversion."""
    m_chunks = BM // 16
    m_reps = BM // 8
    n_per_wave = BN // 4
    num_acc_n = n_per_wave // 16
    store_count = BN // 64
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = lds_typed_ptr(lds_acc_base_i32, T.f32)
    thread = fx.Int32(gpu.thread_id('x'))
    m_lane = thread // fx.Int32(32)
    col_start = thread % fx.Int32(32) * fx.Int32(2)

    def flat_buffer(arg, elem_type, alignment):
        ptr = global_typed_ptr(arg, elem_type, align=alignment)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)
    stids = flat_buffer(arg_stids, T.i32, 4)
    weights = flat_buffer(arg_sweights, T.f32, 4)
    out = flat_buffer(arg_out, T.bf16, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    store_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(cache_modifier=16), fx.BFloat16)

    def load_scalar(atom, src, index, elem_type):
        fragment = fx.make_rmem_tensor(1, elem_type)
        fx.copy(atom, src[None, index], fragment)
        return Vec(fragment.load())[0]
    packed = []
    weight = []
    for mr in range_constexpr(m_reps):
        position = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, position, fx.Int32))
        weight.append(load_scalar(load_f32, weights, position, fx.Float32))
    for i in range_constexpr(m_chunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for j in range_constexpr(num_acc_n):
            column = wave * fx.Int32(n_per_wave) + fx.Int32(j * 16) + lane_mod_16
            vector = Vec(accm[i][j])
            for v in range_constexpr(4):
                index = (row_base + fx.Int32(v)) * fx.Int32(BN) + column
                lds_base[index] = fx.Float32(vector[v])
    gpu.barrier()
    for mr in range_constexpr(m_reps):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token = packed[mr] & fx.Int32(16777215)
        slot = packed[mr] >> fx.Int32(24)
        if (token < i32_M) & (slot < fx.Int32(5)) & (slot >= fx.Int32(0)):
            row_base = ((n_block_idx * i32_M + token) * fx.Int32(5) + slot) * fx.Int32(BN) + col_start
            for s in range_constexpr(store_count):
                index = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                pair = Vec(lds_vec_load(lds_acc_base_i32, index * fx.Int32(4), Vec.make_type(2, fx.Float32), fx.Float32, align=8))
                rounded = Vec.from_elements([pair[0] * weight[mr], pair[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                fragment = fx.make_rmem_tensor(2, fx.BFloat16)
                fragment.store(rounded)
                fx.copy(store_bf16x2, fragment, out[None, row_base + fx.Int32(s * 64)])
_native_body = gemm2._gemm2_body_a16w4
gemm2_body_route_tiled = FunctionType(_native_body.__code__, {**_native_body.__globals__, '_atomic_bf16_epilog': _route_tiled_epilog}, name='gemm2_body_route_tiled', argdefs=_native_body.__defaults__, closure=_native_body.__closure__)
gemm2_body_route_tiled.__kwdefaults__ = _native_body.__kwdefaults__
