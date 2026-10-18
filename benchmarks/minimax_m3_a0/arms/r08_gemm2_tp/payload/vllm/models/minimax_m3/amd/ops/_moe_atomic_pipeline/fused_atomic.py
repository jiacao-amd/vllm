"""Native packed-BF16 atomic GEMM2 directly into the TP reduction workspace."""
from functools import cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels import communication_ops_utils as comm
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import _global_i32_at, _udiv, _umod
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr
from . import layout as cfg
from .atomic_epilog import gemm2_body_atomic_system
from .fused_gemm2 import _exchange, _load_i64, _store_local_i64, _vectorized_rsag

@cache
def compile_atomic(*, column_first=False):
    """One consumer per column; no route tensor and no local merge pass."""

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, cfg.LDS_BYTES, 16]

    @flyc.kernel(name=f'mm3_r08_atomic_system_tp_v1_m96_cm{int(column_first)}', known_block_size=[cfg.THREADS, 1, 1])
    def kernel(a: fx.Int64, weight: fx.Int64, scale: fx.Int64, eids: fx.Int64, counts: fx.Int64, tids: fx.Int64, sorted_weights: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32):
        thread = fx.Int32(gpu.thread_id('x'))
        block = fx.Int32(gpu.block_id('x'))
        blocks = _global_i32_at(counts, fx.Int32(0)) // cfg.BM
        bound = blocks * cfg.COLUMNS
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        if block < bound:
            xcd = _umod(block, 8)
            logical = xcd * _udiv(bound, 8) + fx.Int32(arith.minsi(arith.unwrap(xcd), arith.unwrap(_umod(bound, 8)))) + _udiv(block, 8)
            tile = logical % blocks * cfg.COLUMNS + logical // blocks if const_expr(column_first) else logical
            column = tile % cfg.COLUMNS
            local = _load_i64(peers + fx.Int64(rank) * 8)
            lane = thread % fx.Int32(64)
            wave = fx.rocdl.readfirstlane(T.i32, arith.unwrap(thread // 64))
            gemm2_body_atomic_system(lds, a, weight, scale, eids, tids, sorted_weights, local + cfg.MERGED, tile, lane, wave, fx.Int32(cfg.M), BM=cfg.BM, TILE_N=cfg.TN, TILE_K=cfg.TK, N_OUT=cfg.H, INTER=cfg.INTER, NE=cfg.EXPERTS, b_cache_mod=2, w_dtype='fp4', use_k16=False)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            marker = lds_typed_ptr(fx.Int32(fx.ptrtoint(lds)) + cfg.GEMM_LDS_BYTES, T.i32)
            if thread == fx.Int32(0):
                previous = fx.Int64(comm.atomic_add_agent(local + cfg.COUNTERS + fx.Int64(column) * cfg.FLAG_STRIDE, fx.Int64(1)))
                marker[0] = (previous == fx.Int64(blocks - 1)).select(fx.Int32(1), fx.Int32(0))
                if previous == fx.Int64(blocks - 1):
                    comm.fence_agent_acquire()
            gpu.barrier()
            if marker[0] != fx.Int32(0):
                epoch = _load_i64(local + cfg.EPOCHS + fx.Int64(column) * cfg.FLAG_STRIDE) + 1
                _exchange(peers, local, rank, column, thread, epoch, cfg.MERGE_READY, True)
                _vectorized_rsag(output, peers, local, rank, column, thread, epoch, True)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                gpu.barrier()
                if thread == fx.Int32(0):
                    _store_local_i64(local + cfg.COUNTERS + fx.Int64(column) * cfg.FLAG_STRIDE, fx.Int64(0))
                    _store_local_i64(local + cfg.EPOCHS + fx.Int64(column) * cfg.FLAG_STRIDE, epoch)

    @flyc.jit
    def launch(a: fx.Int64, weight: fx.Int64, scale: fx.Int64, eids: fx.Int64, counts: fx.Int64, tids: fx.Int64, sorted_weights: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel(a, weight, scale, eids, counts, tids, sorted_weights, output, peers, rank).launch(grid=(fx.Int64(grid), 1, 1), block=(cfg.THREADS, 1, 1), stream=stream)
    return launch
