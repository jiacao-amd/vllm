"""Exact-M96 BF16x3 prototype. Import/launch only in the parent's GPU process."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def n18_bf16x3_partials(X, W, P):
    """Six K64 tiles per K384 partition; three independent FP32 accumulators."""
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    split = tl.program_id(2)
    ks = tl.arange(0, 64)
    a0 = tl.zeros((16, 32), tl.float32)
    a1 = tl.zeros((16, 32), tl.float32)
    a2 = tl.zeros((16, 32), tl.float32)
    for tile in range(6):
        k = split * 384 + tile * 64 + ks
        x = tl.load(X + rows[:, None] * 6144 + k[None, :])
        offset = cols[None, :] * 6144 + k[:, None]
        w0 = tl.load(W + offset)
        w1 = tl.load(W + 128 * 6144 + offset)
        w2 = tl.load(W + 2 * 128 * 6144 + offset)
        a0 = tl.dot(x, w0, a0)
        a1 = tl.dot(x, w1, a1)
        a2 = tl.dot(x, w2, a2)
    dst = (split * 3 * 96 + rows[:, None]) * 128 + cols[None, :]
    tl.store(P + dst, a0)
    tl.store(P + dst + 96 * 128, a1)
    tl.store(P + dst + 2 * 96 * 128, a2)


@triton.jit
def n18_reduce_terms(P, Y):
    """Fixed split0..15 order independently per term, then (term0+term1)+term2."""
    row = tl.program_id(0)
    cols = tl.arange(0, 128)
    a0 = tl.zeros((128,), tl.float32)
    a1 = tl.zeros((128,), tl.float32)
    a2 = tl.zeros((128,), tl.float32)
    for split in tl.static_range(16):
        offset = (split * 3 * 96 + row) * 128 + cols
        a0 += tl.load(P + offset)
        a1 += tl.load(P + offset + 96 * 128)
        a2 += tl.load(P + offset + 2 * 96 * 128)
    tl.store(Y + row * 128 + cols, (a0 + a1) + a2)


class Router:
    """Caller-owned 2.25MiB scratch; no allocations, sorting or top-k in call."""

    def __init__(self, split_weight):
        if (tuple(split_weight.shape) != (3, 128, 6144)
                or split_weight.dtype != torch.bfloat16
                or not split_weight.is_contiguous() or not split_weight.is_cuda):
            raise ValueError("N18 requires prepared contiguous BF16[3,128,6144]")
        self.weight = split_weight
        self.partials = torch.empty((16, 3, 96, 128), device=split_weight.device,
                                    dtype=torch.float32)
        self.output = torch.empty((96, 128), device=split_weight.device,
                                  dtype=torch.float32)
        self.compiled = None

    def __call__(self, x):
        if (tuple(x.shape) != (96, 6144) or x.dtype != torch.bfloat16
                or not x.is_contiguous() or x.device != self.weight.device):
            raise ValueError("N18 supports only contiguous BF16[96,6144]")
        self.compiled = n18_bf16x3_partials[(6, 4, 16)](
            x, self.weight, self.partials, num_warps=4, num_stages=2,
            matrix_instr_nonkdim=16, enable_fp_fusion=False)
        n18_reduce_terms[(96,)](self.partials, self.output, num_warps=1,
                                enable_fp_fusion=False)
        return self.output
