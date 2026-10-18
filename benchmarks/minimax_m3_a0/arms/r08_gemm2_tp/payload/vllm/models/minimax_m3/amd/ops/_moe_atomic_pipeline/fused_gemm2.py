"""Natural-grid A16W4 GEMM2 with one last-producer TP consumer per column."""
import os
from functools import cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import communication_ops_utils as comm
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import _global_i32_at, _udiv, _umod
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load
from .route_epilog import gemm2_body_route_bf16
from .route_epilog_tiled import gemm2_body_route_tiled
from . import layout as cfg
MAX_SPINS = 2000000
LOCAL_LOAD_CACHE = 17
INSTRUMENT = os.getenv('MM3_A36_TIMESTAMPS', '0') == '1'

def _load_i64(address):
    return fx.Int64(comm.load_i64_global(address))

def _store_local_i64(address, value):
    """Device-local atomic store; publication to the next call is kernel-ordered."""
    pointer = llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), arith.unwrap(address)).result
    llvm.StoreOp(arith.unwrap(value), pointer, alignment=8, ordering=llvm.AtomicOrdering.monotonic, syncscope=fx.rocdl.SyncScope.AgentOneAs)

@comm.traced
def _stamp_phase(local, column, thread, offset):
    if const_expr(INSTRUMENT):
        if thread == fx.Int32(0):
            tick = llvm.inline_asm(T.i64, [], 's_memrealtime $0\ns_waitcnt lgkmcnt(0)', '=s', has_side_effects=True)
            _store_local_i64(local + offset + fx.Int64(column) * 8, fx.Int64(tick))

def _flat_buffer(address):
    pointer = global_typed_ptr(address, T.bf16, align=16)
    view = fx.Tensor(fx.make_view(pointer, fx.make_layout((1, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=True)

def _load8(buffer, offset, cache_modifier=17):
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=cache_modifier), fx.BFloat16)
    fx.copy(atom, buffer[None, offset], fragment)
    return Vec(fragment.load())

def _store8(buffer, offset, values):
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fragment.store(values)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=16), fx.BFloat16)
    fx.copy(atom, fragment, buffer[None, offset])

@comm.traced
def _exchange(peers, local, rank, column, thread, epoch, phase_offset, native_sync):
    if const_expr(native_sync):
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    else:
        comm.fence_system_release()
    gpu.barrier()
    if const_expr(phase_offset == cfg.MERGE_READY):
        _stamp_phase(local, column, thread, cfg.LOCAL_MERGE_END)
    if thread < fx.Int32(cfg.TP):
        peer_base = _load_i64(peers + fx.Int64(thread) * 8)
        slot = fx.Int64(column) * cfg.FLAG_STRIDE
        comm.store_i64_global_system(peer_base + phase_offset + slot + fx.Int64(rank) * 8, epoch)
        receive = local + phase_offset + slot + fx.Int64(thread) * 8
        current = fx.Int64(comm.load_i64_acquire(receive))
        spins = fx.Int32(0)
        while (current < epoch) & (spins < fx.Int32(MAX_SPINS)):
            fx.rocdl.s_sleep(1)
            current = fx.Int64(comm.load_i64_acquire(receive))
            spins += fx.Int32(1)
        if current < epoch:
            comm.atomic_add_agent(local + cfg.ERROR, fx.Int64(1))
        if const_expr(native_sync):
            comm.fence_system_acquire()
    gpu.barrier()
    if const_expr(not native_sync):
        comm.fence_system_acquire()

@comm.traced
def _vectorized_rsag(output, peers, local, rank, column, thread, epoch, native_sync):
    """All threads reduce their own packs; no peer staging through LDS."""
    peer_ranks = [(rank + offset) % cfg.TP for offset in range_constexpr(cfg.TP)]
    peer_bases = [_load_i64(peers + fx.Int64(peer) * 8) for peer in peer_ranks]
    merged = [_flat_buffer(base + cfg.MERGED) for base in peer_bases]
    reduced = [_flat_buffer(base + cfg.REDUCED) for base in peer_bases]
    own_reduced = _flat_buffer(local + cfg.REDUCED)
    result = _flat_buffer(output)
    part = cfg.M // cfg.TP * (cfg.TN // cfg.PACK)
    for index in range(thread, part, cfg.THREADS):
        item = fx.Int32(index)
        row = item // (cfg.TN // cfg.PACK)
        col = column * cfg.TN + item % (cfg.TN // cfg.PACK) * cfg.PACK
        global_index = (rank * (cfg.M // cfg.TP) + row) * cfg.H + col
        loaded = [_load8(merged[peer], global_index) for peer in range_constexpr(cfg.TP)]
        total = loaded[0].to(fx.Float32)
        for peer in range_constexpr(1, cfg.TP):
            total = total + loaded[peer].to(fx.Float32)
        _store8(own_reduced, row * cfg.H + col, total.to(fx.BFloat16))
    _exchange(peers, local, rank, column, thread, epoch, cfg.REDUCED_READY, native_sync)
    _stamp_phase(local, column, thread, cfg.REDUCE_SYNC_END)
    for index in range(thread, part, cfg.THREADS):
        item = fx.Int32(index)
        row = item // (cfg.TN // cfg.PACK)
        col = column * cfg.TN + item % (cfg.TN // cfg.PACK) * cfg.PACK
        loaded = [_load8(reduced[peer], row * cfg.H + col) for peer in range_constexpr(cfg.TP)]
        for peer in range_constexpr(cfg.TP):
            destination = (peer_ranks[peer] * (cfg.M // cfg.TP) + row) * cfg.H + col
            _store8(result, destination, loaded[peer])

@comm.traced
def _consume_column(lds, routes, output, peers, local, rank, column, thread, native_sync, vectorized_consumer, column_routes):
    _stamp_phase(local, column, thread, cfg.SERVICE_START)
    epoch = _load_i64(local + cfg.EPOCHS + fx.Int64(column) * cfg.FLAG_STRIDE) + 1
    rank_lane = thread // fx.Int32(cfg.LANES_PER_RANK)
    lane = thread % fx.Int32(cfg.LANES_PER_RANK)
    route_buffer = _flat_buffer(routes)
    merged = _flat_buffer(local + cfg.MERGED)
    own_reduced = _flat_buffer(local + cfg.REDUCED)
    output_buffer = _flat_buffer(output)
    part_vectors = cfg.M // cfg.TP * (cfg.TN // cfg.PACK)
    lds_address = fx.Int32(fx.ptrtoint(lds))
    lds_values = lds_typed_ptr(lds_address, T.bf16)
    for item in range(lane, part_vectors, cfg.LANES_PER_RANK):
        item = fx.Int32(item)
        row = item // fx.Int32(cfg.TN // cfg.PACK)
        col = column * cfg.TN + item % fx.Int32(cfg.TN // cfg.PACK) * cfg.PACK
        token = rank_lane * (cfg.M // cfg.TP) + row
        if const_expr(column_routes):
            route_base = (column * cfg.M + token) * cfg.TOPK * cfg.TN + item % fx.Int32(cfg.TN // cfg.PACK) * cfg.PACK
            route_slot_stride = cfg.TN
        else:
            route_base = token * cfg.TOPK * cfg.H + col
            route_slot_stride = cfg.H
        accumulator = Vec.filled(8, 0.0, fx.Float32)
        if const_expr(vectorized_consumer):
            loaded = [_load8(route_buffer, route_base + slot * route_slot_stride, LOCAL_LOAD_CACHE) for slot in range_constexpr(cfg.TOPK)]
            for slot in range_constexpr(cfg.TOPK):
                accumulator = (accumulator + loaded[slot].to(fx.Float32)).to(fx.BFloat16).to(fx.Float32)
        else:
            for slot in range_constexpr(cfg.TOPK):
                value = _load8(route_buffer, route_base + slot * route_slot_stride, LOCAL_LOAD_CACHE)
                accumulator = (accumulator + value.to(fx.Float32)).to(fx.BFloat16).to(fx.Float32)
        _store8(merged, token * cfg.H + col, accumulator.to(fx.BFloat16))
    _exchange(peers, local, rank, column, thread, epoch, cfg.MERGE_READY, native_sync)
    _stamp_phase(local, column, thread, cfg.MERGE_SYNC_END)
    if const_expr(vectorized_consumer):
        _vectorized_rsag(output, peers, local, rank, column, thread, epoch, native_sync)
    else:
        source_rank = (rank + rank_lane) % fx.Int32(cfg.TP)
        peer_base = _load_i64(peers + fx.Int64(source_rank) * 8)
        peer_merged = _flat_buffer(peer_base + cfg.MERGED)
        peer_reduced = _flat_buffer(peer_base + cfg.REDUCED)
        for item in range(lane, part_vectors, cfg.LANES_PER_RANK):
            item = fx.Int32(item)
            row = item // fx.Int32(cfg.TN // cfg.PACK)
            col = column * cfg.TN + item % fx.Int32(cfg.TN // cfg.PACK) * cfg.PACK
            index = (rank * (cfg.M // cfg.TP) + row) * cfg.H + col
            values = _load8(peer_merged, index)
            for element in range_constexpr(cfg.PACK):
                lds_values[thread * cfg.PACK + element] = fx.BFloat16(values[element])
            gpu.barrier()
            if rank_lane == fx.Int32(0):
                summed = Vec(lds_vec_load(lds_address, lane * cfg.PACK * 2, Vec.make_type(8, fx.BFloat16), fx.BFloat16, align=16)).to(fx.Float32)
                for peer in range_constexpr(1, cfg.TP):
                    value = Vec(lds_vec_load(lds_address, (peer * cfg.LANES_PER_RANK + lane) * cfg.PACK * 2, Vec.make_type(8, fx.BFloat16), fx.BFloat16, align=16)).to(fx.Float32)
                    summed = summed + value
                _store8(own_reduced, row * cfg.H + col, summed.to(fx.BFloat16))
            gpu.barrier()
        _exchange(peers, local, rank, column, thread, epoch, cfg.REDUCED_READY, native_sync)
        _stamp_phase(local, column, thread, cfg.REDUCE_SYNC_END)
        for item in range(lane, part_vectors, cfg.LANES_PER_RANK):
            item = fx.Int32(item)
            row = item // fx.Int32(cfg.TN // cfg.PACK)
            col = column * cfg.TN + item % fx.Int32(cfg.TN // cfg.PACK) * cfg.PACK
            values = _load8(peer_reduced, row * cfg.H + col)
            _store8(output_buffer, (source_rank * (cfg.M // cfg.TP) + row) * cfg.H + col, values)
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    gpu.barrier()
    _stamp_phase(local, column, thread, cfg.SERVICE_END)
    if thread == fx.Int32(0):
        counter = local + cfg.COUNTERS + fx.Int64(column) * cfg.FLAG_STRIDE
        _store_local_i64(counter, fx.Int64(0))
        _store_local_i64(local + cfg.EPOCHS + fx.Int64(column) * cfg.FLAG_STRIDE, epoch)

@cache
def compile_fused(*, column_first: bool=True, native_sync: bool=False, vectorized_consumer: bool=False, column_routes: bool=False):
    """One natural-grid GEMM2 dispatch; last producer per column becomes service."""

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, cfg.LDS_BYTES, 16]

    @flyc.kernel(name=f'mm3_gemm2_tp_pipeline_m96_sg1_cm{int(column_first)}_ns{int(native_sync)}_vc{int(vectorized_consumer)}_cr{int(column_routes)}_lr{LOCAL_LOAD_CACHE}_clock{int(INSTRUMENT)}', known_block_size=[cfg.THREADS, 1, 1])
    def kernel(a: fx.Int64, weight: fx.Int64, scale: fx.Int64, eids: fx.Int64, counts: fx.Int64, tids: fx.Int64, sorted_weights: fx.Int64, routes: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32):
        thread = fx.Int32(gpu.thread_id('x'))
        block = fx.Int32(gpu.block_id('x'))
        blocks = _global_i32_at(counts, fx.Int32(0)) // cfg.BM
        bound = blocks * cfg.COLUMNS
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        if block < bound:
            xcd = _umod(block, 8)
            logical = xcd * _udiv(bound, 8) + fx.Int32(arith.minsi(arith.unwrap(xcd), arith.unwrap(_umod(bound, 8)))) + _udiv(block, 8)
            if const_expr(column_first):
                tile = logical % blocks * cfg.COLUMNS + logical // blocks
            else:
                tile = logical
            column = tile % cfg.COLUMNS
            lane = thread % fx.Int32(64)
            wave = fx.rocdl.readfirstlane(T.i32, arith.unwrap(thread // 64))
            body = gemm2_body_route_tiled if column_routes else gemm2_body_route_bf16
            body(lds, a, weight, scale, eids, tids, sorted_weights, routes, tile, lane, wave, fx.Int32(cfg.M), BM=cfg.BM, TILE_N=cfg.TN, TILE_K=cfg.TK, N_OUT=cfg.H, INTER=cfg.INTER, NE=cfg.EXPERTS, b_cache_mod=2, w_dtype='fp4', use_k16=False)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            local = _load_i64(peers + fx.Int64(rank) * 8)
            marker = lds_typed_ptr(fx.Int32(fx.ptrtoint(lds)) + cfg.GEMM_LDS_BYTES, T.i32)
            if thread == fx.Int32(0):
                previous = fx.Int64(comm.atomic_add_agent(local + cfg.COUNTERS + fx.Int64(column) * cfg.FLAG_STRIDE, fx.Int64(1)))
                marker[0] = (previous == fx.Int64(blocks - 1)).select(fx.Int32(1), fx.Int32(0))
                if previous == fx.Int64(blocks - 1):
                    comm.fence_agent_acquire()
            gpu.barrier()
            if marker[0] != fx.Int32(0):
                _consume_column(lds, routes, output, peers, local, rank, column, thread, native_sync, vectorized_consumer, column_routes)

    @flyc.jit
    def launch(a: fx.Int64, weight: fx.Int64, scale: fx.Int64, eids: fx.Int64, counts: fx.Int64, tids: fx.Int64, sorted_weights: fx.Int64, routes: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel(a, weight, scale, eids, counts, tids, sorted_weights, routes, output, peers, rank).launch(grid=(fx.Int64(grid), 1, 1), block=(cfg.THREADS, 1, 1), stream=stream)
    return launch

@cache
def compile_consumer_only(*, native_sync: bool=False, vectorized_consumer: bool=False, column_routes: bool=False):
    """Diagnostic consumer kernel; prime with active_columns=0 before TP replay."""

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, cfg.LDS_BYTES, 16]

    @flyc.kernel(name=f'mm3_a36_consumer_diagnostic_ns{int(native_sync)}_vc{int(vectorized_consumer)}_cr{int(column_routes)}_lr{LOCAL_LOAD_CACHE}_clock{int(INSTRUMENT)}', known_block_size=[cfg.THREADS, 1, 1])
    def kernel(routes: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32, active_columns: fx.Int32):
        column = fx.Int32(gpu.block_id('x'))
        thread = fx.Int32(gpu.thread_id('x'))
        if column < active_columns:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
            local = _load_i64(peers + fx.Int64(rank) * 8)
            _consume_column(lds, routes, output, peers, local, rank, column, thread, native_sync, vectorized_consumer, column_routes)

    @flyc.jit
    def launch(routes: fx.Int64, output: fx.Int64, peers: fx.Int64, rank: fx.Int32, active_columns: fx.Int32, stream: fx.Stream):
        kernel(routes, output, peers, rank, active_columns).launch(grid=(cfg.COLUMNS, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream)
    return launch
