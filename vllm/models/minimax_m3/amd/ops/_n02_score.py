# SPDX-License-Identifier: Apache-2.0
"""N02: retain native decode-score code; bound only the C24/Q4/M96 grid."""

from contextvars import ContextVar
from functools import cache
import hashlib
import json
import math
import os
from pathlib import Path
from types import FunctionType

CAP = 432
SOURCE_SHA256 = "34a11711c49a0724df833c447a08e3529dc64a37373d9ade8927c1fe41431dfa"
_audit_bound = ContextVar("n02_score_max_seq_len", default=None)


def bounded_split(native_split, max_blocks: int, requests: int) -> tuple[int, int]:
    """Exactly A41 cap432: retain native waves and never increase native chunks."""
    chunks, waves = native_split(max_blocks, requests)
    return min(chunks, max(1, CAP // max(1, requests))), waves


def supported(query, cache_tensor, lengths, query_len: int) -> bool:
    """Use only host tensor metadata; native code retains all dtype/layout checks."""
    return (
        tuple(query.shape) == (96, 1, 128)
        and query_len == 4
        and lengths.numel() == 24
        and tuple(cache_tensor.shape[1:]) == (128, 128)
    )


@cache
def implementation(audit: bool = False):
    """Clone native Python code with a private splitter; never monkey-patch AITER."""
    from aiter.ops import msa_block_select as native

    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("N02 native score source differs from the pinned nightly")
    original = native.pa_sparse_block_score_decode
    scope = dict(original.__globals__)
    native_split = scope["_score_split"]
    scope["_score_split"] = lambda blocks, requests: bounded_split(native_split, blocks, requests)
    if audit:
        native_launch = scope["_launch_score_decode"]
        reported = set()

        def audited_launch(*args, **kwargs):
            import torch.distributed as dist

            bound = _audit_bound.get()
            if bound is None:
                raise RuntimeError("Launch audit must enter through N02 score_decode")
            grid = (kwargs["num_q_tiles"], kwargs["num_reqs"], kwargs["num_chunks"])
            key = (grid, bound)
            if key not in reported:
                native_chunks, native_waves = native_split(
                    math.ceil(bound / kwargs["block_size"]), kwargs["num_reqs"]
                )
                print("MM3_N02_SCORE_LAUNCH " + json.dumps({
                    "rank": dist.get_rank(), "grid": grid,
                    "native_grid": (grid[0], grid[1], native_chunks),
                    "block_threads": kwargs["num_waves"] * 64,
                    "native_block_threads": native_waves * 64,
                    "query_len": kwargs["query_len"], "max_seq_len": bound, "cap": CAP,
                }), flush=True)
                reported.add(key)
            return native_launch(*args, **kwargs)

        scope["_launch_score_decode"] = audited_launch
    result = FunctionType(original.__code__, scope, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def score_decode(
    q_idx, key_cache_idx, score, block_table, seq_lens,
    init_blocks=0, local_blocks=0, query_len=1, max_seq_len=0,
):
    """Keep native ABI and fallback, including all caller-supplied sentinels/bounds."""
    from aiter.ops.msa_block_select import pa_sparse_block_score_decode

    args = (q_idx, key_cache_idx, score, block_table, seq_lens,
            init_blocks, local_blocks, query_len, max_seq_len)
    if not supported(q_idx, key_cache_idx, seq_lens, query_len):
        return pa_sparse_block_score_decode(*args)
    audit = os.getenv("MM3_N02_SCORE_AUDIT", "0") == "1"
    if not audit:
        return implementation()(*args)
    token = _audit_bound.set(max_seq_len)
    try:
        return implementation(True)(*args)
    finally:
        _audit_bound.reset(token)
