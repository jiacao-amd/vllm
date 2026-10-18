"""Shape-guarded A41 dispatch inside MiniMax's existing opaque attention op."""

import os

from .score_dispatch import score_decode

SELECTED_CAP = 432


def dispatch(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    init_blocks=0,
    local_blocks=0,
    query_len=1,
    max_seq_len=0,
):
    if os.getenv("MM3_INDEXER_SCORE_GRID_EXPERIMENT", "0") == "1":
        return score_decode(
            q_idx,
            key_cache_idx,
            score,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            query_len,
            max_seq_len,
            cap=SELECTED_CAP,
        )
    from aiter.ops.msa_block_select import pa_sparse_block_score_decode

    return pa_sparse_block_score_decode(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        seq_lens,
        init_blocks,
        local_blocks,
        query_len,
        max_seq_len,
    )
