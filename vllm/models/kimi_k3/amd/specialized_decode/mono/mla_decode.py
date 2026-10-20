# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from the MFMA32 forward in pytorch/pytorch#194309.
"""gfx950 compressed MLA decode, derived from PR194309's MFMA32 forward.

Conc1: Q[Sq,H,576], token-indexed FP8 KV[N,576], O[Sq,H,512].
The index table maps rank-local logical tokens to physical cache slots. Global
sequence length includes the verify block; DCP is token round robin. Both output
and natural-log LSE are returned for vLLM's cross-rank attention merge.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def _global(tensor, size):
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(fx.get_iter(tensor), fx.make_layout(size, 1))
    )


def _value_layout(rows, cols):
    return fx.make_layout(
        ((8, rows // 8), (32, cols // 32)), ((32, 8 * cols), (1, 256))
    )


def _exp(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


@functools.lru_cache(None)
def _build(*, max_local, cache_size, cp_rank, return_lse=True):
    heads = 96
    qlen = 8
    splits = 42
    cp_world = 8
    waves = 4
    value_tile = 512
    q_fp8 = True
    kv_fp8 = True
    pv_fp8 = True
    native = True
    prob_scale = 256.0
    scale = 192 ** (-0.5)
    rows = heads * qlen
    threads = 64 * waves
    chunks = value_tile // 32
    kv_type = fx.Float8E4M3FN
    query_blocks = (rows + waves * 32 - 1) // (waves * 32)
    vblocks = 512 // value_tile
    native = q_fp8 and kv_fp8
    pv_fp8 = native and pv_fp8
    vtype = fx.Float8E4M3FN
    tile_kv = 64
    shared_elements = tile_kv * 576 // 2 * 2

    @fx.struct
    class Shared:
        kv: fx.Array[fx.BFloat16, shared_elements, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def stage1(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        seq_len: fx.Tensor,
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
        partial: fx.Tensor,
        stats: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        split = fx.Int32(fx.block_idx.y)
        vbase = fx.Int32(fx.block_idx.z) * value_tile
        lds = fx.SharedAllocator().allocate(Shared).peek()
        qbuf = _global(q, rows * 576)
        kvbuf = _global(kv, cache_size * 576)
        ibuf = _global(indices, max_local)
        sbuf = _global(seq_len, 1)
        pbuf = _global(partial, splits * rows * 512)
        stbuf = _global(stats, splits * rows * 2)
        qs = fx.Float32(_global(q_scale, 1)[0])
        ks = fx.Float32(_global(kv_scale, 1)[0])
        score_scale = qs * ks * fx.Float32(scale * math.log2(math.e))
        glen = fx.Int32(sbuf[0])
        raw_len = (glen + cp_world - 1 - cp_rank) // cp_world
        local_len = (raw_len > 0).select(raw_len, fx.Int32(0))
        tile_count = (local_len + tile_kv - 1) // tile_kv
        active_splits = (tile_count < 42).select(tile_count, fx.Int32(42))
        active_splits = (active_splits > 0).select(active_splits, fx.Int32(1))
        split_tiles = (local_len + active_splits * tile_kv - 1) // (
            active_splits * tile_kv
        )
        begin = split * split_tiles
        last = (local_len + tile_kv - 1) // tile_kv
        end = (begin + split_tiles < last).select(begin + split_tiles, last)
        mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr = mma.get_slice(lane)
        ccoord = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (1, 0))))
        rcoord = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (0, 1))))
        row = (
            fx.Int32(fx.block_idx.x) * (waves * 32)
            + wave * 32
            + fx.Int32(fx.get_scalar(rcoord[0]))
        )
        safe_row = (row < rows).select(row, fx.Int32(0))
        raw_limit = (glen - qlen + safe_row // heads + cp_world - cp_rank) // cp_world
        limit = (raw_limit > 0).select(raw_limit, fx.Int32(0))
        klayout = _value_layout(tile_kv, 576)
        kptr = fx.recast_iter(fx.Float8E4M3FN, lds.kv.ptr)
        vlayout = _value_layout(tile_kv, 576)
        v_mma_layout = fx.composition(
            fx.select(vlayout, [1, 0]),
            fx.make_tile(
                fx.make_layout(576, 1),
                fx.make_layout((4, 2, 2, tile_kv // 16), (1, 8, 4, 16)),
            ),
        )
        value_tiles = fx.flat_divide(fx.make_view(kptr, v_mma_layout), (32, 16))
        transpose = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(8, 64), vtype)
        copy_v = fx.make_tiled_copy_A(transpose, mma).get_slice(lane)
        qregs = []
        cp8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
        for step in fx.range_constexpr(9):
            words = []
            for pack in fx.range_constexpr(2):
                frag = fx.make_rmem_tensor(16, fx.Float8E4M3FN)
                col = step * 64 + lane // 32 * 32 + pack * 16
                source = fx.make_view(
                    fx.get_iter(qbuf) + safe_row * 576 + col, fx.make_layout(16, 1)
                )
                fx.copy(cp8, source, frag)
                packed = Vec(frag.load()).bitcast(fx.Int32)
                words.extend([packed[i] for i in fx.range_constexpr(4)])
            qregs.append(Vec.from_elements(words, fx.Int32))

        def prefetch_key(base):
            coords = fx.right_inverse(klayout)
            slots, columns, keeps, frags = ([], [], [], [])
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                logical = fx.Int32(
                    fx.get_scalar(fx.crd2idx((it * threads + tid) * 16, coords))
                )
                token = base + logical % tile_kv
                keep = token < local_len
                slots.append(fx.Int32(ibuf[keep.select(token, fx.Int32(0))]))
                columns.append(logical // tile_kv)
                keeps.append(keep)
            fx.rocdl.sched_barrier(0)
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                frag = fx.make_rmem_tensor(16, kv_type)
                src = fx.make_view(
                    fx.get_iter(kvbuf) + slots[it] * 576 + columns[it],
                    fx.make_layout(16, 1),
                )
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy128b(), kv_type), src, frag
                )
                frags.append(frag)
            fx.rocdl.sched_barrier(0)
            return (frags, keeps)

        def store_prefetched(frags, keeps, ptr):
            dst = fx.logical_divide(
                fx.make_view(ptr, fx.make_layout(tile_kv * 576, 1)),
                fx.make_layout(16, 1),
            )
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                words = Vec(frags[it].load()).bitcast(fx.Int32)
                clean = Vec.from_elements(
                    [
                        keeps[it].select(fx.Int32(words[j]), fx.Int32(0))
                        for j in fx.range_constexpr(4)
                    ],
                    fx.Int32,
                )
                frags[it].store(clean.bitcast(kv_type))
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy128b(), kv_type),
                    frags[it],
                    fx.slice(dst, (None, it * threads + tid)),
                )

        zero = Vec.filled(16, 0.0, fx.Float32)
        initial = [fx.Float32(-1e30), fx.Float32(0.0)] + [zero for _ in range(chunks)]
        final = initial
        preload, keep_preload = prefetch_key(begin * tile_kv)
        store_prefetched(preload, keep_preload, kptr)
        fx.gpu.barrier()
        # FlyDSL lowers this loop with loop-carried SSA values.
        for tile, args in range(  # type: ignore[call-overload]
            begin, end, fx.Int32(1), init=initial
        ):
            tile = fx.Int32(tile)
            oldmax, oldsum = (fx.Float32(args[0]), fx.Float32(args[1]))
            accum = [args[2 + d] for d in fx.range_constexpr(chunks)]
            stage = (tile - begin) % 2
            baseptr = fx.recast_iter(fx.Float8E4M3FN, lds.kv.ptr)
            kptr = baseptr + stage * (tile_kv * 576)
            nextptr = baseptr + (1 - stage) * (tile_kv * 576)
            value_tiles = fx.flat_divide(fx.make_view(kptr, v_mma_layout), (32, 16))
            prefetched, keep_next = prefetch_key((tile + 1) * tile_kv)
            tile_scores = [zero, zero]
            cp8lds = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN)
            for step in fx.range_constexpr(9):
                qr = Vec(qregs[step])
                for half in fx.range_constexpr(2):
                    shared_row = fx.logical_divide(
                        fx.slice(
                            fx.make_view(kptr, klayout), (lane % 32 + half * 32, None)
                        ),
                        fx.make_layout(16, 1),
                    )
                    words = []
                    for pack in fx.range_constexpr(2):
                        frag = fx.make_rmem_tensor(16, fx.Float8E4M3FN)
                        fx.copy(
                            cp8lds,
                            fx.slice(
                                shared_row, (None, step * 4 + lane // 32 * 2 + pack)
                            ),
                            frag,
                        )
                        packed = Vec(frag.load()).bitcast(fx.Int32)
                        words.extend([packed[i] for i in fx.range_constexpr(4)])
                    a = Vec.from_elements(words, fx.Int32)
                    tile_scores[half] = Vec(
                        fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                            T.vec(16, T.f32),
                            [
                                a.ir_value(),
                                qr.ir_value(),
                                Vec(tile_scores[half]).ir_value(),
                                0,
                                0,
                                0,
                                fx.Int32(2139062143).ir_value(),
                                0,
                                fx.Int32(2139062143).ir_value(),
                            ],
                        )
                    )
            fx.rocdl.sched_dsrd(8)
            for step in fx.range_constexpr(9):
                fx.rocdl.sched_mfma(2)
                if const_expr(step + 2 < 9):
                    fx.rocdl.sched_dsrd(4)
            fx.rocdl.sched_barrier(0)
            values, valid = ([], [])
            for half in fx.range_constexpr(2):
                for e in fx.range_constexpr(16):
                    keep = (row < rows) & (
                        tile * tile_kv + half * 32 + fx.Int32(fx.get_scalar(ccoord[e]))
                        < limit
                    )
                    valid.append(keep)
                    values.append(
                        keep.select(
                            fx.Float32(Vec(tile_scores[half])[e]) * score_scale,
                            fx.Float32(float("-inf")),
                        )
                    )
            levels = values
            for depth in fx.range_constexpr(5):
                levels = [
                    levels[2 * i].maximumf(levels[2 * i + 1])
                    for i in fx.range_constexpr(16 >> depth)
                ]
            m = (
                levels[0]
                .maximumf(fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64)))
                .maximumf(oldmax)
            )
            correction = _exp(oldmax - m)
            probs = [_exp(values[i] - m) for i in fx.range_constexpr(32)]
            levels = probs
            for depth in fx.range_constexpr(5):
                levels = [
                    levels[2 * i] + levels[2 * i + 1]
                    for i in fx.range_constexpr(16 >> depth)
                ]
            total = (
                oldsum * correction
                + levels[0]
                + fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64))
            )
            corr = Vec.from_elements([correction], fx.Float32).broadcast_to(16)
            words = []
            for w in fx.range_constexpr(8):
                off = w * 4
                word = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probs[off] * prob_scale,
                    probs[off + 1] * prob_scale,
                    fx.Int32(0),
                    0,
                )
                word = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probs[off + 2] * prob_scale,
                    probs[off + 3] * prob_scale,
                    word,
                    1,
                )
                words.append(fx.Int32(word))
            bp = Vec.from_elements(words, fx.Int32)
            for d in fx.range_constexpr(chunks):
                vwords = []
                for pack in fx.range_constexpr(4):
                    vt = fx.slice(value_tiles, (None, None, d + vbase // 32, pack))
                    a = fx.make_fragment_like(thr.partition_A(vt), vtype)
                    fx.copy(transpose, copy_v.partition_S(vt), copy_v.retile(a))
                    packed = Vec(a.load()).bitcast(fx.Int32)
                    vwords.extend([packed[0], packed[1]])
                ap = Vec.from_elements(vwords, fx.Int32)
                scaled_acc = Vec(accum[d]) * corr
                accum[d] = fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                    T.vec(16, T.f32),
                    [
                        ap.ir_value(),
                        bp.ir_value(),
                        scaled_acc.ir_value(),
                        0,
                        0,
                        0,
                        fx.Int32(2139062143).ir_value(),
                        0,
                        fx.Int32(2139062143).ir_value(),
                    ],
                )
                fx.rocdl.sched_barrier(0)
            fx.rocdl.sched_dsrd(8)
            for d in fx.range_constexpr(chunks):
                fx.rocdl.sched_mfma(1)
                if const_expr(d + 2 < chunks):
                    fx.rocdl.sched_dsrd(4)
            fx.rocdl.sched_barrier(0)
            store_prefetched(prefetched, keep_next, nextptr)
            fx.gpu.barrier()
            final = yield [m, total, *accum]
        if row < rows:
            for d in fx.range_constexpr(chunks):
                vals = Vec(final[2 + d]).bitcast(fx.Int32)
                for group in fx.range_constexpr(2):
                    lower, upper = ([], [])
                    for i in fx.range_constexpr(4):
                        pair = fx.rocdl.permlane32_swap(
                            ir.Type.parse("!llvm.struct<(i32, i32)>"),
                            fx.Int32(vals[group * 8 + i]).ir_value(),
                            fx.Int32(vals[group * 8 + i + 4]).ir_value(),
                            False,
                            True,
                        )
                        lower.append(fx.Int32(llvm.extractvalue(T.i32, pair, [0])))
                        upper.append(fx.Int32(llvm.extractvalue(T.i32, pair, [1])))
                    col = vbase + d * 32 + group * 16 + lane // 32 * 8
                    frag = fx.make_rmem_tensor(8, fx.BFloat16)
                    frag.store(
                        Vec.from_elements(lower + upper, fx.Int32)
                        .bitcast(fx.Float32)
                        .to(fx.BFloat16)
                    )
                    dst = fx.make_view(
                        fx.get_iter(pbuf) + (split * rows + row) * 512 + col,
                        fx.make_layout(8, 1),
                    )
                    fx.copy(
                        fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
                        frag,
                        dst,
                    )
            if (lane < 32) & (fx.block_idx.z == 0):
                stbuf[(split * rows + row) * 2] = final[0]
                stbuf[(split * rows + row) * 2 + 1] = final[1]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def stage2(
        partial: fx.Tensor,
        stats: fx.Tensor,
        kv_scale: fx.Tensor,
        seq_len: fx.Tensor,
        output: fx.Tensor,
        lse: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        lane = fx.Int32(fx.thread_idx.x)
        pb = _global(partial, splits * rows * 512)
        sb = _global(stats, splits * rows * 2)
        ob = _global(output, rows * 512)
        if const_expr(return_lse):
            lb = _global(lse, rows)
        ks = fx.Float32(_global(kv_scale, 1)[0])

        def combine(count):
            maxima, sums = ([], [])
            m = fx.Float32(-1e30)
            for s in fx.range_constexpr(count):
                mx = fx.Float32(sb[(s * rows + row) * 2])
                sm = fx.Float32(sb[(s * rows + row) * 2 + 1])
                maxima.append(mx)
                sums.append(sm)
                m = m.maximumf(mx)
            weights = [_exp(maxima[s] - m) for s in fx.range_constexpr(count)]
            total = fx.Float32(0.0)
            acc = Vec.filled(8, 0.0, fx.Float32)
            for s in fx.range_constexpr(count):
                total = total + sums[s] * weights[s]
                part = fx.make_rmem_tensor(8, fx.BFloat16)
                src = fx.make_view(
                    fx.get_iter(pb) + (s * rows + row) * 512 + lane * 8,
                    fx.make_layout(8, 1),
                )
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), src, part
                )
                pv = Vec(part.load()).to(fx.Float32)
                acc = acc + pv * Vec.from_elements(
                    [weights[s]], fx.Float32
                ).broadcast_to(8)
            return (acc, m, total)

        acc = Vec.filled(8, 0.0, fx.Float32)
        m, total = (fx.Float32(-1e30), fx.Float32(0.0))
        glen = fx.Int32(_global(seq_len, 1)[0])
        if glen <= 8192 + cp_rank:
            acc, m, total = combine(16)
        elif glen <= 16384 + cp_rank:
            acc, m, total = combine(32)
        else:
            acc, m, total = combine(42)
        inv = (total > 0.0).select(ks / (total * prob_scale), fx.Float32(0.0))
        frag = fx.make_rmem_tensor(8, fx.BFloat16)
        frag.store(
            (acc * Vec.from_elements([inv], fx.Float32).broadcast_to(8)).to(fx.BFloat16)
        )
        dst = fx.make_view(fx.get_iter(ob) + row * 512 + lane * 8, fx.make_layout(8, 1))
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), frag, dst)
        if const_expr(return_lse):  # noqa: SIM102 - separate compile-time guard
            if lane == 0:
                lb[row] = (total > 0.0).select(
                    (m + fx.math.log2(total)) * math.log(2.0), fx.Float32(float("-inf"))
                )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        seq_len: fx.Tensor,
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
        partial: fx.Tensor,
        stats: fx.Tensor,
        output: fx.Tensor,
        lse: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL ABI default
    ):
        stage1(q, kv, indices, seq_len, q_scale, kv_scale, partial, stats).launch(
            grid=(query_blocks, splits, vblocks), block=(threads, 1, 1), stream=stream
        )
        stage2(partial, stats, kv_scale, seq_len, output, lse).launch(
            grid=(rows, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch


class KimiMLADecodePlan:
    """Reuse one plan on a single stream or for calls ordered by stream events.

    Q is FP8 E4M3FN [8, 96, 576]. KV is FP8 E4M3FN [cache_slots, 576].
    The int32 index vector maps logical rank-local tokens to physical KV slots.
    global_seq_len is a device int32 scalar, including the eight verify tokens.
    Its value must be >= 8 and its rank-local length must fit max_local_tokens.
    q_scale and kv_scale are device FP32 dequantization multipliers.

    O is BF16 [8, 96, 512]; LSE is FP32 [8, 96], in natural-log units.
    DCP callers need LSE to merge rank-local outputs. No max-score output exists.
    The launcher allocates no tensors, reads no device scalars on the host, and
    uses device-side split selection, including during CUDA Graph replay.
    Warm up once before capturing a graph. Outputs are overwritten by each call.
    """

    def __init__(
        self,
        *,
        max_local_tokens: int,
        cache_slots: int,
        cp_rank: int,
        device: torch.device | str = "cuda",
        return_lse: bool = True,
        workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if not 0 <= cp_rank < 8:
            raise ValueError("cp_rank must be in [0, 8)")
        if max_local_tokens < 1 or cache_slots < 1:
            raise ValueError("KV capacities must be positive")
        if cache_slots * 576 >= 2**31:
            raise ValueError("KV storage must fit the signed 32-bit element offsets")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Kimi MLA requires a gfx950 GPU")
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise ValueError(f"Kimi MLA requires gfx950, got {arch}")
        self.device = device
        self.max_local_tokens = max_local_tokens
        self.cache_slots = cache_slots
        self.return_lse = return_lse
        self.output = torch.empty((8, 96, 512), dtype=torch.bfloat16, device=device)
        self.lse = (
            torch.empty((8, 96), dtype=torch.float32, device=device)
            if return_lse
            else None
        )
        self._unused_lse = (
            torch.empty((1,), dtype=torch.float32, device=device)
            if not return_lse
            else None
        )
        if workspace is None:
            self.partial = torch.empty(
                (42, 8, 96, 512), dtype=torch.bfloat16, device=device
            )
            self.stats = torch.empty((42, 8, 96, 2), dtype=torch.float32, device=device)
        else:
            self.partial, self.stats = workspace
            self._check(
                "partial workspace", self.partial, (42, 8, 96, 512), torch.bfloat16
            )
            self._check("stats workspace", self.stats, (42, 8, 96, 2), torch.float32)
        self._launch = _build(
            max_local=max_local_tokens,
            cache_size=cache_slots,
            cp_rank=cp_rank,
            return_lse=return_lse,
        )

    def _check(self, name, tensor, shape, dtype):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {self.device}")

    def __call__(
        self,
        q,
        kv,
        indices,
        global_seq_len,
        q_scale,
        kv_scale,
        *,
        output=None,
        lse=None,
        stream=None,
    ):
        self._check("q", q, (8, 96, 576), torch.float8_e4m3fn)
        self._check("kv", kv, (self.cache_slots, 576), torch.float8_e4m3fn)
        self._check("indices", indices, (self.max_local_tokens,), torch.int32)
        self._check("global_seq_len", global_seq_len, (1,), torch.int32)
        # vLLM scale tensors may be zero-dimensional; both forms are contiguous.
        for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if scale.numel() != 1 or scale.ndim > 1:
                raise ValueError(f"{name} must be a scalar or one-element vector")
            self._check(name, scale, scale.shape, torch.float32)
        # FlyDSL's automatic tensor wrapping requires a stride-1 axis.
        # These metadata-only views also accept vLLM's zero-dimensional scales.
        if q_scale.ndim == 0:
            q_scale = q_scale.view(1)
        if kv_scale.ndim == 0:
            kv_scale = kv_scale.view(1)
        output = self.output if output is None else output
        self._check("output", output, (8, 96, 512), torch.bfloat16)
        if self.return_lse:
            lse = self.lse if lse is None else lse
            self._check("lse", lse, (8, 96), torch.float32)
        elif lse is not None:
            raise ValueError("This plan was built with return_lse=False")
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self._launch(
            q,
            kv,
            indices,
            global_seq_len,
            q_scale,
            kv_scale,
            self.partial,
            self.stats,
            output,
            lse if self.return_lse else self._unused_lse,
            stream=stream,
        )
        return output, lse
