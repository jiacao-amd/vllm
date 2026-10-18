"""Fixed M3 TP4 workspace geometry, shared by CPU checks and the GPU emitter."""
M = 96
H = 6144
INTER = 768
EXPERTS = 129
TOPK = 5
TP = 4
BM = 16
TN = 256
TK = 256
COLUMNS = H // TN
THREADS = 256
LANES_PER_RANK = THREADS // TP
PACK = 8
FLAG_STRIDE = 128
COUNTERS = 0
EPOCHS = COUNTERS + COLUMNS * FLAG_STRIDE
MERGE_READY = EPOCHS + COLUMNS * FLAG_STRIDE
REDUCED_READY = MERGE_READY + COLUMNS * FLAG_STRIDE
ERROR = REDUCED_READY + COLUMNS * FLAG_STRIDE
SERVICE_START = ERROR + 128
MERGE_SYNC_END = SERVICE_START + 256
REDUCE_SYNC_END = MERGE_SYNC_END + 256
SERVICE_END = REDUCE_SYNC_END + 256
LOCAL_MERGE_END = SERVICE_END + 256
META_BYTES = 16384
MERGED = META_BYTES
REDUCED = MERGED + M * H * 2
USED_BYTES = REDUCED + M // TP * H * 2
ALLOCATION_BYTES = 2 * 1024 * 1024
GEMM_LDS_BYTES = BM * TK * 2 + BM * TN * 4
LDS_BYTES = GEMM_LDS_BYTES + 32

def tile_index(pid: int, blocks: int, column_first: bool) -> int:
    """Native XCD remap, optionally transposed to finish columns earlier."""
    count = blocks * COLUMNS
    xcd = pid % 8
    logical = xcd * (count // 8) + min(xcd, count % 8) + pid // 8
    if column_first:
        return logical % blocks * COLUMNS + logical // blocks
    return logical

def consumer_indices(column: int, rank: int, thread: int) -> list[tuple[int, int, int]]:
    """Return packed (local-merge, reduced-shard, gathered-output) indices."""
    rank_lane, lane = divmod(thread, LANES_PER_RANK)
    part = M // TP * (TN // PACK)
    result = []
    for item in range(lane, part, LANES_PER_RANK):
        row, col_vec = divmod(item, TN // PACK)
        shard = row * (H // PACK) + column * (TN // PACK) + col_vec
        merged = rank_lane * (M // TP) * (H // PACK) + shard
        gathered = (rank + rank_lane) % TP * (M // TP) * (H // PACK) + shard
        result.append((merged, shard, gathered))
    return result
