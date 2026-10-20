# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AITER persistent CPRR baseline with vLLM native-head padding."""

import aiter
import torch
from aiter.mla import mla_decode_fwd


def make_aiter(data, *, world=8, rank=0, splits=16):
    q, kv, idx, seq, qs, ks, ref, ref_lse = data
    qlen, heads = q.shape[:2]
    padded_heads = 128 if heads > 64 else 16
    padded = torch.empty((qlen, padded_heads, 576), dtype=q.dtype, device=q.device)
    padded.view(torch.uint8)[:, :heads] = q.view(torch.uint8)
    padded.view(torch.uint8)[:, heads:] = q.view(torch.uint8)[:, : padded_heads - heads]
    length = seq.item()
    local = max(0, (length + world - 1 - rank) // world)
    qi = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    ki = torch.tensor([0, local], dtype=torch.int32, device="cuda")
    gi = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    last = torch.ones(1, dtype=torch.int32, device="cuda")
    info = aiter.get_mla_metadata_info_v1(
        1,
        qlen,
        padded_heads,
        q.dtype,
        kv.dtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=splits,
        intra_batch_mode=False,
    )
    meta = [torch.empty(size, dtype=dtype, device="cuda") for size, dtype in info]
    wm, wi, ws, ri, rf, rp = meta
    aiter.get_mla_metadata_v1(
        qi,
        ki,
        last,
        padded_heads,
        1,
        False,
        wm,
        ws,
        wi,
        ri,
        rf,
        rp,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=splits,
        intra_batch_mode=False,
        dtype_q_nope=q.dtype,
        dtype_kv_nope=kv.dtype,
        is_cp_round_robin=world > 1,
    )
    out = torch.zeros((qlen, padded_heads, 512), device="cuda", dtype=torch.bfloat16)

    def run():
        return mla_decode_fwd(
            padded,
            kv.view(-1, 1, 1, 576),
            out,
            qi,
            ki,
            idx[:local],
            last,
            max_seqlen_q=qlen,
            sm_scale=192**-0.5,
            num_kv_splits=splits,
            q_scale=qs,
            kv_scale=ks,
            work_meta_data=wm,
            work_indptr=wi,
            work_info_set=ws,
            reduce_indptr=ri,
            reduce_final_map=rf,
            reduce_partial_map=rp,
            g_kv_indptr=gi,
            cp_world_size=world,
            cp_rank=rank,
            return_lse=True,
        )

    return run, out
