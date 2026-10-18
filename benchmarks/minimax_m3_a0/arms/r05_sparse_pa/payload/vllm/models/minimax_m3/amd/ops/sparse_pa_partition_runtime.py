# SPDX-License-Identifier: Apache-2.0
"""Private, shape-guarded MiniMax-M3 sparse PA partition experiment."""

import os

import torch


def eligible(query, key, value, table, kv_heads, splits, key_scale, value_scale):
    """Only the measured TP4/M96/page16/scalar-FP8 contract may use the kernel."""
    return (
        tuple(query.shape) == (96, 16, 128)
        and query.dtype == torch.bfloat16
        and kv_heads == 1
        and splits == 6
        and tuple(table.shape) == (96, 128)
        and table.dtype == torch.int32
        and key.dtype == value.dtype == torch.float8_e4m3fn
        and key.ndim == value.ndim == 5
        and tuple(key.shape[1:]) == (1, 8, 16, 16)
        and tuple(value.shape[1:]) == (1, 1, 128, 16)
        and key.device == value.device == query.device == table.device
        and all(
            scale is not None
            and scale.numel() == 1
            and scale.dtype == torch.float32
            and scale.device == query.device
            for scale in (key_scale, value_scale)
        )
    )


def select_decode(query, key, value, table, kv_heads, splits, key_scale, value_scale):
    """The opaque attention operation keeps the same output/metadata contract."""
    if os.getenv("MM3_SPARSE_PA_PARTITION_EXPERIMENT", "0") != "1":
        return None
    if not eligible(query, key, value, table, kv_heads, splits, key_scale, value_scale):
        return None
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return None
    from .sparse_pa_partition import load

    return load(hybrid=True).pa_decode_gluon
