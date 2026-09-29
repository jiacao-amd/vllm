# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER backend for MiMo's asymmetric FP8 KV cache."""

from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, MultipleOf
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadata,
    AiterFlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

_FULL_DECODE_WORKSPACE_ATTR = "_aiter_diffkv_full_decode_workspace"

if current_platform.is_rocm():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _reshape_and_cache_diffkv_shuffle_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slot_mapping_ptr,
        k_scale_ptr,
        v_scale_ptr,
        stride_key_token: tl.int64,
        stride_key_head: tl.int64,
        stride_value_token: tl.int64,
        stride_value_head: tl.int64,
        stride_key_cache_block: tl.int64,
        stride_value_cache_block: tl.int64,
        block_size: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        head_size_v: tl.constexpr,
        x: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
        if slot_idx < 0:
            return

        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size
        k_offsets = tl.arange(0, BLOCK_K)
        v_offsets = tl.arange(0, BLOCK_V)

        k = tl.load(
            key_ptr
            + token_idx * stride_key_token
            + head_idx * stride_key_head
            + k_offsets,
            mask=k_offsets < head_size,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            value_ptr
            + token_idx * stride_value_token
            + head_idx * stride_value_head
            + v_offsets,
            mask=v_offsets < head_size_v,
            other=0.0,
        ).to(tl.float32)

        k = k / tl.load(k_scale_ptr)
        v = v / tl.load(v_scale_ptr)

        k_dst = (
            key_cache_ptr
            + block_idx * stride_key_cache_block
            + head_idx * head_size * block_size
            + (k_offsets // x) * block_size * x
            + block_offset * x
            + k_offsets % x
        )
        v_dst = (
            value_cache_ptr
            + block_idx * stride_value_cache_block
            + head_idx * head_size_v * block_size
            + (block_offset // x) * head_size_v * x
            + v_offsets * x
            + block_offset % x
        )
        tl.store(k_dst, k, mask=k_offsets < head_size)
        tl.store(v_dst, v, mask=v_offsets < head_size_v)


def _split_diffkv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the padded K/V planes as logical NHD tensors."""
    if kv_cache.ndim != 4 or kv_cache.shape[1] != 2:
        raise ValueError(
            "AITER DiffKV cache must have logical shape [B, 2, N, C], "
            f"got {tuple(kv_cache.shape)}"
        )
    num_blocks, _, block_size, side_width = kv_cache.shape
    key_width = num_kv_heads * head_size
    value_width = num_kv_heads * head_size_v
    if side_width < max(key_width, value_width):
        raise ValueError(
            "AITER DiffKV cache side is too small: "
            f"{side_width} < {max(key_width, value_width)}"
        )

    key_plane = kv_cache[:, 0]
    value_plane = kv_cache[:, 1]
    key_cache = torch.as_strided(
        key_plane,
        size=(num_blocks, block_size, num_kv_heads, head_size),
        stride=(key_plane.stride(0), key_width, head_size, 1),
        storage_offset=key_plane.storage_offset(),
    )
    value_cache = torch.as_strided(
        value_plane,
        size=(num_blocks, block_size, num_kv_heads, head_size_v),
        stride=(value_plane.stride(0), value_width, head_size_v, 1),
        storage_offset=value_plane.storage_offset(),
    )
    return key_cache, value_cache


def _native_diffkv_cache_views(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reinterpret raw K/V planes in the layouts consumed by FlyDSL PA."""
    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    value_head_size = value_cache.shape[-1]
    x = 16 // key_cache.element_size()
    key_native = torch.as_strided(
        key_cache,
        size=(num_blocks, num_kv_heads, head_size // x, block_size, x),
        stride=(
            key_cache.stride(0),
            head_size * block_size,
            block_size * x,
            x,
            1,
        ),
        storage_offset=key_cache.storage_offset(),
    )
    value_native = torch.as_strided(
        value_cache,
        size=(
            num_blocks,
            num_kv_heads,
            block_size // x,
            value_head_size,
            x,
        ),
        stride=(
            value_cache.stride(0),
            value_head_size * block_size,
            value_head_size * x,
            x,
            1,
        ),
        storage_offset=value_cache.storage_offset(),
    )
    return key_native, value_native


def reshape_and_cache_diffkv_shuffle(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize K/V with static scales and write FlyDSL shuffle layouts."""
    if not current_platform.is_rocm():
        raise RuntimeError("AITER DiffKV cache update requires ROCm")
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    block_size = key_cache.shape[1]
    x = 16 // key_cache.element_size()
    _reshape_and_cache_diffkv_shuffle_kernel[(num_tokens, num_kv_heads)](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache.stride(0),
        value_cache.stride(0),
        block_size,
        num_kv_heads,
        head_size,
        head_size_v,
        x,
        BLOCK_K=triton.next_power_of_2(head_size),
        BLOCK_V=triton.next_power_of_2(head_size_v),
        num_warps=8,
    )


class AiterDiffKVAttentionBackend(AiterFlashAttentionBackend):
    """MiMo QK192/V128 backend with AITER prefill and FlyDSL decode."""

    head_size_v: int = 128
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]
    forward_includes_kv_cache_update = False

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["AiterDiffKVAttentionImpl"]:
        return AiterDiffKVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AiterFlashAttentionMetadataBuilder]:
        return AiterFlashAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 192

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.block_size not in (1, 16):
            raise ValueError("AITER DiffKV requires block size 16.")
        if spec.head_size != 192 or spec.head_size_v != cls.head_size_v:
            raise ValueError(
                "AITER DiffKV requires Q/K head size 192 and V head size "
                f"{cls.head_size_v}."
            )
        side_width = spec.num_kv_heads * max(spec.head_size, spec.head_size_v)
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=side_width * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)


class AiterDiffKVAttentionImpl(AiterFlashAttentionImpl):
    """Run MiMo decode with FlyDSL while retaining AITER prefill support."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_size_v = AiterDiffKVAttentionBackend.head_size_v
        if (self.head_size, self.head_size_v) != (192, 128):
            raise NotImplementedError(
                "AITER DiffKV currently supports Q/K=192 and V=128 only."
            )
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("AITER DiffKV requires an FP8 KV cache.")
        if self.alibi_slopes is not None or self.logits_soft_cap != 0.0:
            raise NotImplementedError(
                "AITER DiffKV does not support ALiBi or logits soft cap."
            )
        self._decode_workspace: dict[tuple[int, torch.dtype, int], tuple] = {}

    def _kv_cache_layout_name(self) -> str:
        return "SHUFFLE"

    def _uses_metadata_kv_scales(self) -> bool:
        return False

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _split_diffkv_cache(
            kv_cache,
            self.num_kv_heads,
            self.head_size,
            self.head_size_v,
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        fp8_dtype = current_platform.fp8_dtype()
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        reshape_and_cache_diffkv_shuffle(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_rope_kvcache_supported(self):
        return False

    def _get_full_decode_workspace(
        self,
        context_lengths: torch.Tensor,
        query: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
    ):
        from aiter.ops.flydsl.pa_decode import plan_pa_decode

        shared = getattr(attn_metadata, _FULL_DECODE_WORKSPACE_ATTR, None)
        if shared is not None:
            return shared

        batch_size = context_lengths.numel()
        device_index = query.device.index
        key = (batch_size, query.dtype, -1 if device_index is None else device_index)
        cached = self._decode_workspace.get(key)
        if cached is None:
            plan = plan_pa_decode(context_lengths, self.num_kv_heads)
            rows = self.num_heads // self.num_kv_heads
            scalar_shape = (self.num_kv_heads, plan.capacity, rows)
            exp_sums = torch.empty(
                scalar_shape, dtype=torch.float32, device=query.device
            )
            max_logits = torch.empty_like(exp_sums)
            temporary_output = torch.empty(
                (*scalar_shape, self.head_size_v),
                dtype=query.dtype,
                device=query.device,
            )
            cached = (plan, exp_sums, max_logits, temporary_output)
            self._decode_workspace[key] = cached
        else:
            plan, _, _, _ = cached
            plan_pa_decode(
                context_lengths,
                self.num_kv_heads,
                plan=plan,
            )
        setattr(attn_metadata, _FULL_DECODE_WORKSPACE_ATTR, cached)
        return cached

    def _forward_decode(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        from aiter.ops.flydsl.pa_decode import pa_decode

        assert attn_metadata.decode_metadata is not None
        query_length = attn_metadata.decode_metadata.uniform_query_len
        if query_length != 1:
            raise NotImplementedError(
                "AITER DiffKV currently requires uniform single-token decode."
            )

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        fp8_dtype = current_platform.fp8_dtype()
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        key_cache, value_cache = _native_diffkv_cache_views(key_cache, value_cache)
        context_lengths = attn_metadata.seq_lens[:num_decodes]
        block_tables = attn_metadata.block_table[:num_decodes]
        sliding_window = (
            self.sliding_window[0] + 1 if self.sliding_window[0] >= 0 else 0
        )

        kwargs = {}
        max_partitions = 1
        if sliding_window == 0:
            plan, exp_sums, max_logits, temporary_output = (
                self._get_full_decode_workspace(
                    context_lengths,
                    query,
                    attn_metadata,
                )
            )
            max_partitions = plan.max_partitions
            kwargs = {
                "work_plan": plan,
                "exp_sums": exp_sums,
                "max_logits": max_logits,
                "temporary_output": temporary_output,
            }
        elif sliding_window != 128:
            raise NotImplementedError(
                "AITER DiffKV currently supports only SWA128 or full attention."
            )

        pa_decode(
            output=output[:num_decode_tokens],
            query=query[:num_decode_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=context_lengths,
            block_tables=block_tables,
            softmax_scale=self.scale,
            query_length=1,
            max_context_partition_num=max_partitions,
            compute_type=fp8_dtype,
            key_scale=layer._k_scale,
            value_scale=layer._v_scale,
            sinks=self.sinks,
            sliding_window=sliding_window,
            **kwargs,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "AITER DiffKV does not support fused output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.use_cascade:
            raise NotImplementedError(
                "AITER DiffKV does not support cascade attention."
            )

        num_decode_tokens = attn_metadata.num_decode_tokens
        if attn_metadata.num_decodes > 0:
            self._forward_decode(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
            )

        remaining_tokens = attn_metadata.num_actual_tokens - num_decode_tokens
        if remaining_tokens == 0:
            return output

        suffix_query_start = attn_metadata.query_start_loc[attn_metadata.num_decodes :]
        suffix_query_start = suffix_query_start - suffix_query_start[0]
        suffix_metadata = replace(
            attn_metadata,
            num_actual_tokens=remaining_tokens,
            query_start_loc=suffix_query_start,
            seq_lens=attn_metadata.seq_lens[attn_metadata.num_decodes :],
            slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:],
            block_table=attn_metadata.block_table[attn_metadata.num_decodes :],
            num_decodes=0,
            num_decode_tokens=0,
            decode_metadata=None,
        )
        suffix = slice(num_decode_tokens, attn_metadata.num_actual_tokens)
        super().forward(
            layer,
            query[suffix],
            None if key is None else key[suffix],
            None if value is None else value[suffix],
            kv_cache,
            suffix_metadata,
            output[suffix],
        )
        return output
