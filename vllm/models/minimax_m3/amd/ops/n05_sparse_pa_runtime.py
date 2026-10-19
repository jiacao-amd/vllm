# SPDX-License-Identifier: Apache-2.0
"""Metadata-only selection for the fixed TP4/M96 single-KV-head decode lane."""

import os


def eligible(
    query,
    key,
    value,
    table,
    lengths,
    output,
    kv_heads,
    splits,
    key_scale,
    value_scale,
    softmax_scale,
):
    """No tensor-value reads or synchronization; every other layout stays native."""
    tensors = (query, key, value, table, lengths, output)
    return (
        tuple(query.shape) == (96, 16, 128)
        and tuple(output.shape) == tuple(query.shape)
        and str(query.dtype) == str(output.dtype) == "torch.bfloat16"
        and query.device.type == "cuda"
        and all(tensor.device == query.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and kv_heads == 1
        and splits == 6
        and softmax_scale == 128**-0.5
        and tuple(table.shape) == (96, 128)
        and tuple(lengths.shape) == (96,)
        and str(table.dtype) == str(lengths.dtype) == "torch.int32"
        and str(key.dtype) == str(value.dtype) == "torch.float8_e4m3fn"
        and key.ndim == value.ndim == 5
        and key.shape[0] > 0
        and value.shape[0] > 0
        and tuple(key.shape[1:]) == (1, 8, 16, 16)
        and tuple(value.shape[1:]) == (1, 1, 128, 16)
        and key.shape[0] - value.shape[0] in (0, 8)
        and all(
            scale is not None
            and scale.numel() == 1
            and str(scale.dtype) == "torch.float32"
            and scale.device == query.device
            and scale.is_contiguous()
            for scale in (key_scale, value_scale)
        )
    )


def select_decode(
    query,
    key,
    value,
    table,
    lengths,
    output,
    kv_heads,
    splits,
    key_scale,
    value_scale,
    softmax_scale,
):
    """Only the four-token decode caller may enter this optional private path."""
    if os.getenv("MM3_SPARSE_PA_PARTITION_EXPERIMENT", "0") != "1":
        return None
    if not eligible(
        query,
        key,
        value,
        table,
        lengths,
        output,
        kv_heads,
        splits,
        key_scale,
        value_scale,
        softmax_scale,
    ):
        return None
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.platforms.rocm import on_gfx950

    if get_tensor_model_parallel_world_size() != 4 or not on_gfx950():
        return None
    from .n05_sparse_pa_partition import load

    return load().pa_decode_gluon
