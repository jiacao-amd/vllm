# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""K3 inputs and FP32 attention reference, including token-round-robin DCP."""

import torch


def reference(q, kv, indices, qs, ks, length, world, rank):
    qlen, heads, _ = q.shape
    local = max(0, (length + world - 1 - rank) // world)
    logical = kv.float()[indices[:local].long()]
    score = (q.float().reshape(qlen * heads, 576) * qs) @ (logical * ks).T * (192**-0.5)
    limit = torch.tensor(
        [max(0, (length - qlen + t + world - rank) // world) for t in range(qlen)],
        device="cuda",
    ).repeat_interleave(heads)
    score.masked_fill_(
        torch.arange(local, device="cuda")[None, :] >= limit[:, None], -float("inf")
    )
    lse = score.logsumexp(-1).reshape(qlen, heads)
    prob = score.softmax(-1).nan_to_num()
    ref = (prob @ (logical[:, :512] * ks)).reshape(qlen, heads, 512)
    return ref, lse


def make_case(
    length,
    heads,
    qlen,
    world,
    rank,
    q_fp8=True,
    kv_fp8=True,
    seed=1234,
    layout="random",
    magnitude=0.5,
    capacity=None,
):
    torch.manual_seed(seed + length)
    local = max(0, (length + world - 1 - rank) // world)
    capacity = max(capacity or 0, 64, (local + 63) // 64 * 64)
    q = (torch.randn((qlen, heads, 576), device="cuda") * magnitude).to(
        torch.float8_e4m3fn if q_fp8 else torch.bfloat16
    )
    logical = (torch.randn((capacity, 576), device="cuda") * magnitude).to(
        torch.float8_e4m3fn if kv_fp8 else torch.bfloat16
    )
    # NoPE skips the rotation, but all 576 learned channels still contribute.
    if layout == "random":
        indices = torch.randperm(capacity, device="cuda").int()
    elif layout == "paged":
        indices = (
            (
                torch.randperm(capacity // 64, device="cuda")[:, None] * 64
                + torch.arange(64, device="cuda")
            )
            .flatten()
            .int()
        )
    else:
        indices = torch.arange(capacity, device="cuda", dtype=torch.int32)
    kv = torch.empty_like(logical)
    kv.view(torch.uint8)[indices.long()] = logical.view(torch.uint8)
    seqlen = torch.tensor([length], dtype=torch.int32, device="cuda")
    qs = torch.tensor([0.75 if q_fp8 else 1.0], device="cuda")
    ks = torch.tensor([1.2 if kv_fp8 else 1.0], device="cuda")
    ref, lse = reference(q, kv, indices, qs, ks, length, world, rank)
    return q, kv, indices, seqlen, qs, ks, ref, lse
