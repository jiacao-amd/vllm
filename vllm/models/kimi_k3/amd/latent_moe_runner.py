# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = init_logger(__name__)

_KIMI_K3_ROUTED_HIDDEN = 3584
_KIMI_K3_SHARED_HIDDEN = 7168
_KIMI_K3_INTERMEDIATE_PER_TP_RANK = 384
_KIMI_K3_SHARED_INTERMEDIATE_PER_TP_RANK = 768
_KIMI_K3_TOPK = 16
_KIMI_K3_FHMOE_MAX_BATCH = 32


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner, but currently only the up projection
    -sharded path is implemented. (Tier 2)

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        tp_size = self.moe_config.tp_size

        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and up_proj.weight.shape[0] % tp_size == 0
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = up_proj.weight.shape[0] // tp_size
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False
        self._kimi_k3_fhmoe_mxfp4_enabled = envs.VLLM_ROCM_KIMI_K3_FHMOE_MXFP4
        self._kimi_k3_fhmoe_initialized = False
        self._kimi_k3_fhmoe_available = False
        self._kimi_k3_fhmoe_routed_w1: torch.Tensor | None = None
        self._kimi_k3_fhmoe_routed_w2: torch.Tensor | None = None
        self._kimi_k3_fhmoe_routed_w1_scale: torch.Tensor | None = None
        self._kimi_k3_fhmoe_routed_w2_scale: torch.Tensor | None = None
        self._kimi_k3_fhmoe_shared_w1: torch.Tensor | None = None
        self._kimi_k3_fhmoe_shared_w2: torch.Tensor | None = None
        self._kimi_k3_fhmoe_shared_out: torch.Tensor | None = None
        self._kimi_k3_fhmoe_workspace = None
        self._kimi_k3_fhmoe_op = None
        self._logged_kimi_k3_fhmoe_mxfp4 = False

    def _initialize_kimi_k3_fhmoe_mxfp4(self) -> bool:
        """Validate and cache the fixed Kimi-K3 TP8 contract once per layer."""

        if self._kimi_k3_fhmoe_initialized:
            return self._kimi_k3_fhmoe_available
        self._kimi_k3_fhmoe_initialized = True

        if (
            not self._kimi_k3_fhmoe_mxfp4_enabled
            or self.enable_dbo
            or self._shared_experts is None
            or self.moe_config.tp_size != 8
            or self.moe_config.dp_size != 1
            or self.moe_config.ep_size != 1
            or self.moe_config.is_sequence_parallel
            or self.moe_config.experts_per_token != _KIMI_K3_TOPK
            or self.moe_config.hidden_dim != _KIMI_K3_ROUTED_HIDDEN
            or self.moe_config.intermediate_size_per_partition
            != _KIMI_K3_INTERMEDIATE_PER_TP_RANK
            or self.routed_experts.activation != MoEActivation.SITU
        ):
            return False

        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False

        self.routed_experts._ensure_moe_quant_config_init()
        quant_method = self._quant_method
        backend_name = getattr(
            getattr(quant_method, "mxfp4_backend", None),
            "name",
            None,
        )
        quant_config = getattr(quant_method, "moe_quant_config", None)
        if (
            quant_method.is_monolithic
            or backend_name != "AITER_MXFP4_BF16"
            or quant_method.moe_kernel is None
            or quant_config is None
            or not quant_config.use_mxfp4_w4a16
        ):
            return False

        routed_w1 = getattr(self.routed_experts, "w13_weight", None)
        routed_w2 = getattr(self.routed_experts, "w2_weight", None)
        routed_w1_scale = getattr(self.routed_experts, "w13_weight_scale", None)
        routed_w2_scale = getattr(self.routed_experts, "w2_weight_scale", None)
        routed_down_proj = self.routed_input_transform
        shared_layer = getattr(self._shared_experts, "_layer", None)
        gate_up_proj = getattr(shared_layer, "gate_up_proj", None)
        down_proj = getattr(shared_layer, "down_proj", None)
        routed_down_w = getattr(routed_down_proj, "weight", None)
        shared_w1 = getattr(gate_up_proj, "weight", None)
        shared_w2 = getattr(down_proj, "weight", None)
        if not all(
            isinstance(weight, torch.Tensor)
            for weight in (
                routed_w1,
                routed_w2,
                routed_w1_scale,
                routed_w2_scale,
                routed_down_w,
                shared_w1,
                shared_w2,
            )
        ):
            return False

        assert isinstance(routed_w1, torch.Tensor)
        assert isinstance(routed_w2, torch.Tensor)
        assert isinstance(routed_w1_scale, torch.Tensor)
        assert isinstance(routed_w2_scale, torch.Tensor)
        assert isinstance(routed_down_w, torch.Tensor)
        assert isinstance(shared_w1, torch.Tensor)
        assert isinstance(shared_w2, torch.Tensor)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        expert_count = routed_w1.shape[0] if routed_w1.ndim == 3 else 0
        if (
            fp4_dtype is None
            or routed_w1.dtype != fp4_dtype
            or routed_w2.dtype != fp4_dtype
            or not getattr(routed_w1, "is_shuffled", False)
            or not getattr(routed_w2, "is_shuffled", False)
            or tuple(routed_w1.shape)
            != (
                expert_count,
                2 * _KIMI_K3_INTERMEDIATE_PER_TP_RANK,
                _KIMI_K3_ROUTED_HIDDEN // 2,
            )
            or tuple(routed_w2.shape)
            != (
                expert_count,
                _KIMI_K3_ROUTED_HIDDEN,
                _KIMI_K3_INTERMEDIATE_PER_TP_RANK // 2,
            )
            or expert_count < _KIMI_K3_TOPK
            or tuple(routed_down_w.shape)
            != (_KIMI_K3_ROUTED_HIDDEN, _KIMI_K3_SHARED_HIDDEN)
            or tuple(shared_w1.shape)
            != (
                2 * _KIMI_K3_SHARED_INTERMEDIATE_PER_TP_RANK,
                _KIMI_K3_SHARED_HIDDEN,
            )
            or tuple(shared_w2.shape)
            != (
                _KIMI_K3_SHARED_HIDDEN,
                _KIMI_K3_SHARED_INTERMEDIATE_PER_TP_RANK,
            )
            or routed_down_w.dtype != torch.bfloat16
            or shared_w1.dtype != torch.bfloat16
            or shared_w2.dtype != torch.bfloat16
            or len(
                {
                    routed_w1.device,
                    routed_w2.device,
                    routed_w1_scale.device,
                    routed_w2_scale.device,
                    routed_down_w.device,
                    shared_w1.device,
                    shared_w2.device,
                }
            )
            != 1
            or any(
                not weight.is_cuda or not weight.is_contiguous()
                for weight in (
                    routed_w1,
                    routed_w2,
                    routed_w1_scale,
                    routed_w2_scale,
                    routed_down_w,
                    shared_w1,
                    shared_w2,
                )
            )
        ):
            return False

        try:
            from aiter.ops.flydsl.kimi_k3_fhmoe import (
                create_kimi_k3_fhmoe_workspace,
                kimi_k3_fhmoe_a16w4,
                prepare_kimi_k3_fhmoe_shared_weights,
            )
        except ImportError:
            logger.warning_once(
                "Kimi-K3 MXFP4 FHMoE was requested, but this AITER build "
                "does not provide the heterogeneous two-stage kernel. "
                "Falling back.",
                scope="global",
            )
            return False

        shared_w1_shuffled, shared_w2_shuffled = prepare_kimi_k3_fhmoe_shared_weights(
            shared_w1.detach(),
            shared_w2.detach(),
        )
        max_sorted_tokens = (
            _KIMI_K3_FHMOE_MAX_BATCH * _KIMI_K3_TOPK + expert_count * 32 - _KIMI_K3_TOPK
        )
        device = routed_w1.device
        self._kimi_k3_fhmoe_workspace = create_kimi_k3_fhmoe_workspace(
            max_sorted_tokens=max_sorted_tokens,
            max_tokens=_KIMI_K3_FHMOE_MAX_BATCH,
            num_experts=expert_count,
            device=device,
        )
        self._kimi_k3_fhmoe_shared_out = torch.empty(
            (_KIMI_K3_FHMOE_MAX_BATCH, _KIMI_K3_SHARED_HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        self._kimi_k3_fhmoe_routed_w1 = routed_w1
        self._kimi_k3_fhmoe_routed_w2 = routed_w2
        self._kimi_k3_fhmoe_routed_w1_scale = routed_w1_scale
        self._kimi_k3_fhmoe_routed_w2_scale = routed_w2_scale
        self._kimi_k3_fhmoe_shared_w1 = shared_w1_shuffled
        self._kimi_k3_fhmoe_shared_w2 = shared_w2_shuffled
        self._kimi_k3_fhmoe_op = kimi_k3_fhmoe_a16w4
        self._kimi_k3_fhmoe_available = True
        return True

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
    ):
        num_tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        if (
            not self._kimi_k3_fhmoe_mxfp4_enabled
            or not 1 <= num_tokens <= _KIMI_K3_FHMOE_MAX_BATCH
            or shared_experts_input is None
            or shared_experts_overlapping
            or hidden_states.dtype != torch.bfloat16
            or shared_experts_input.dtype != torch.bfloat16
            or tuple(hidden_states.shape) != (num_tokens, _KIMI_K3_ROUTED_HIDDEN)
            or tuple(shared_experts_input.shape) != (num_tokens, _KIMI_K3_SHARED_HIDDEN)
            or not self._initialize_kimi_k3_fhmoe_mxfp4()
        ):
            return super()._apply_quant_method(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                shared_experts_overlapping,
            )

        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=self._quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
        assert self._kimi_k3_fhmoe_op is not None
        assert self._kimi_k3_fhmoe_workspace is not None
        assert self._kimi_k3_fhmoe_shared_out is not None
        assert self._kimi_k3_fhmoe_routed_w1 is not None
        assert self._kimi_k3_fhmoe_routed_w2 is not None
        assert self._kimi_k3_fhmoe_routed_w1_scale is not None
        assert self._kimi_k3_fhmoe_routed_w2_scale is not None
        assert self._kimi_k3_fhmoe_shared_w1 is not None
        assert self._kimi_k3_fhmoe_shared_w2 is not None
        fused_out, shared_out = self._kimi_k3_fhmoe_op(
            routed_x=hidden_states,
            shared_x=shared_experts_input,
            routed_w1=self._kimi_k3_fhmoe_routed_w1,
            routed_w2=self._kimi_k3_fhmoe_routed_w2,
            routed_w1_scale=self._kimi_k3_fhmoe_routed_w1_scale,
            routed_w2_scale=self._kimi_k3_fhmoe_routed_w2_scale,
            shared_w1=self._kimi_k3_fhmoe_shared_w1,
            shared_w2=self._kimi_k3_fhmoe_shared_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=self._kimi_k3_fhmoe_workspace,
            shared_out=self._kimi_k3_fhmoe_shared_out[:num_tokens],
        )

        if not self._logged_kimi_k3_fhmoe_mxfp4:
            self._logged_kimi_k3_fhmoe_mxfp4 = True
            logger.info_once(
                "Using experimental Kimi-K3 M=1..32 heterogeneous FHMoE: "
                "one Stage-1 launch and one Stage-2 launch for routed MXFP4 "
                "experts plus the BF16 shared experts.",
                scope="global",
            )
        return shared_out, fused_out

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """
        Tier 2: column-parallel up-projection folded into the final reduce.
        """
        if not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        up_proj_shard = transform.up_proj.weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tail_shardable and not self._fused_output_is_reduced:
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        result = self._shard_up_proj_tail(
            fused_output, shared_output, og_hidden_dim_post_xform
        )

        return self._maybe_add_zero_expert_output(result)
