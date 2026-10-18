# SPDX-License-Identifier: Apache-2.0
# Adapted from AITER/FlyDSL's A16W4 BF16 atomic epilogue.
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Native BF16 atomic arithmetic with system-coherent publication to IPC memory."""
from types import FunctionType
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from . import gemm2_slot as gemm2
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import _raw
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load

@flyc.jit
def _system_atomic_copy(arg_out, out_off, pk):
    """Device-local-only packed BF16 atomic with explicit system coherence."""
    address = arg_out + fx.Int64(out_off) * fx.Int64(2)
    llvm.inline_asm(None, [arith.unwrap(address), _raw(pk.bitcast(fx.Int32)[0])], 'global_atomic_pk_add_bf16 $0, $1, off sc1', 'v,v,~{memory}', has_side_effects=True)

@flyc.jit
def _atomic_system_epilog(lds_acc_base_i32, accm, arg_out, arg_stids, arg_sweights, m_row, n_block_idx, wave, lane, i32_M, BM, N_OUT, BN):
    _kMChunks = BM // 16
    M_REPS = BM // 8
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base_fptr = lds_typed_ptr(lds_acc_base_i32, T.f32)
    tx_i32 = fx.Int32(gpu.thread_id('x'))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)

    def _flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)
    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = _flat_buffer(arg_out, T.bf16, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))
        weight.append(load_scalar(load_f32, sweights, sorted_pos, fx.Float32))
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                lds_base_fptr[idx] = fx.Float32(vec[v])
    gpu.barrier()
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(16777215)
        if token_id < i32_M:
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(lds_vec_load(lds_acc_base_i32, idx0 * fx.Int32(4), Vec.make_type(2, fx.Float32), fx.Float32, align=8))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                out_frag = fx.make_rmem_tensor(2, fx.BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int32(s * 64)
                _system_atomic_copy(arg_out, out_off, pk)
_native_body = gemm2._gemm2_body_a16w4
gemm2_body_atomic_system = FunctionType(_native_body.__code__, {**_native_body.__globals__, '_atomic_bf16_epilog': _atomic_system_epilog}, name='gemm2_body_atomic_system', argdefs=_native_body.__defaults__, closure=_native_body.__closure__)
gemm2_body_atomic_system.__kwdefaults__ = _native_body.__kwdefaults__
