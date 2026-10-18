"""Private native score callable with a bounded grid; no GPU math changes."""

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from types import FunctionType

try:
    from .policy import SOURCE_SHA256, bounded_split
except ImportError:
    from policy import SOURCE_SHA256, bounded_split


@cache
def implementation(cap):
    from aiter.ops import msa_block_select as native

    source = Path(native.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Native score launch source differs from audited A39")
    original = native.pa_sparse_block_score_decode
    scope = dict(original.__globals__)
    native_split = scope["_score_split"]
    scope["_score_split"] = lambda blocks, requests: bounded_split(
        native_split, blocks, requests, cap
    )
    if os.getenv("MM3_A41_LAUNCH_AUDIT", "0") == "1":
        original_launch = scope["_launch_score_decode"]
        reported = set()

        def audited_launch(*args, **kwargs):
            grid = (kwargs["num_q_tiles"], kwargs["num_reqs"], kwargs["num_chunks"])
            if grid not in reported:
                import torch.distributed as dist

                print(
                    "MM3_A41_LAUNCH "
                    + json.dumps(
                        {
                            "rank": dist.get_rank(),
                            "grid": grid,
                            "block_threads": kwargs["num_waves"] * 64,
                            "cap": cap,
                            "query_len": kwargs["query_len"],
                        }
                    ),
                    flush=True,
                )
                reported.add(grid)
            return original_launch(*args, **kwargs)

        scope["_launch_score_decode"] = audited_launch
    candidate = FunctionType(
        original.__code__,
        scope,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    candidate.__kwdefaults__ = original.__kwdefaults__
    return candidate


def score_decode(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    init_blocks=0,
    local_blocks=0,
    query_len=1,
    max_seq_len=0,
    *,
    cap=384,
):
    """Only the measured TP4/C24 query geometry uses the bounded launch."""
    from aiter.ops.msa_block_select import pa_sparse_block_score_decode

    supported = (
        tuple(q_idx.shape) == (96, 1, 128)
        and query_len == 4
        and seq_lens.numel() == 24
        and tuple(key_cache_idx.shape[1:]) == (128, 128)
    )
    launch = implementation(cap) if supported else pa_sparse_block_score_decode
    return launch(
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
