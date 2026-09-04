# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Kimi-K3's TP8 small-decode heterogeneous FHMoE.

The baseline runs the native routed and shared expert paths. The experimental
path uses one Stage-1 launch and one Stage-2 launch for the routed MXFP4 and
shared BF16 experts.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiMoE
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072
TOPK = 16
SHARED_EXPERTS = 2
DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--samples", type=int, default=7)
    return parser.parse_args()


def make_layer(
    device: torch.device,
    rank: int,
    world_size: int,
    num_experts: int,
) -> tuple[KimiMoE, VllmConfig]:
    config = KimiLinearConfig(
        hidden_size=HIDDEN_SIZE,
        hidden_act="situ",
        rms_norm_eps=1e-5,
        moe_intermediate_size=INTERMEDIATE_SIZE,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_experts=num_experts,
        num_experts_per_token=TOPK,
        num_shared_experts=SHARED_EXPERTS,
        routed_scaling_factor=1.0,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routed_expert_hidden_size=LATENT_SIZE,
    )
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size),
        kernel_config=KernelConfig(moe_backend="aiter"),
    )

    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(DTYPE),
    ):
        layer = KimiMoE(
            config,
            quant_config=Mxfp4Config(),
            prefix="model.layers.1.mlp",
            layer_idx=1,
        ).to(device)

        generator = torch.Generator(device=device).manual_seed(20260903 + rank)
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if parameter.dtype == torch.uint8:
                    if name.endswith("weight_scale"):
                        parameter.fill_(120)
                    else:
                        parameter.random_(0, 256, generator=generator)
                elif parameter is layer.gate.e_score_correction_bias:
                    parameter.zero_()
                elif parameter is layer.routed_expert_norm.weight:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.01, generator=generator)

        layer.experts._quant_method.process_weights_after_loading(
            layer.experts.routed_experts
        )

    return layer.eval(), vllm_config


@torch.inference_mode()
def run_forward(
    layer: KimiMoE,
    vllm_config: VllmConfig,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=hidden_states.shape[0],
    ):
        return layer(hidden_states)


@torch.inference_mode()
def measure(
    layer: KimiMoE,
    vllm_config: VllmConfig,
    inputs: list[torch.Tensor],
    warmup: int,
    iterations: int,
    samples: int,
) -> list[float]:
    for index in range(warmup):
        run_forward(layer, vllm_config, inputs[index % len(inputs)])
    torch.accelerator.synchronize()

    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        dist.barrier()
        start.record()
        for iteration in range(iterations):
            run_forward(
                layer,
                vllm_config,
                inputs[iteration % len(inputs)],
            )
        end.record()
        end.synchronize()
        latency = torch.tensor(
            start.elapsed_time(end) * 1000.0 / iterations,
            dtype=torch.float64,
            device=inputs[0].device,
        )
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        samples_us.append(float(latency))
    return samples_us


def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return float(error / scale)


def worker(rank: int, world_size: int, port: int, args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size),
        kernel_config=KernelConfig(moe_backend="aiter"),
    )

    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=f"tcp://localhost:{port}",
                local_rank=rank,
            )
            ensure_model_parallel_initialized(world_size, 1)
            init_workspace_manager(device)
            layer, vllm_config = make_layer(
                device,
                rank,
                world_size,
                args.num_experts,
            )

            generator = torch.Generator(device=device).manual_seed(20260904)
            inputs = [
                torch.randn(
                    (args.batch_size, HIDDEN_SIZE),
                    generator=generator,
                    dtype=DTYPE,
                    device=device,
                )
                * 0.02
                for _ in range(args.rotations)
            ]
            runner = layer.experts
            assert isinstance(runner, ROCmLatentMoERunner)
            assert runner._shared_experts is not None

            runner._kimi_k3_fhmoe_mxfp4_enabled = False
            reference = run_forward(layer, vllm_config, inputs[0])
            baseline_us = measure(
                layer,
                vllm_config,
                inputs,
                args.warmup,
                args.iterations,
                args.samples,
            )

            runner._kimi_k3_fhmoe_mxfp4_enabled = True
            optimized_actual = run_forward(layer, vllm_config, inputs[0])
            optimized_us = measure(
                layer,
                vllm_config,
                inputs,
                args.warmup,
                args.iterations,
                args.samples,
            )

            # Re-run the native path after the fused weight storage has been
            # installed. This separates a real execution gain from ordering,
            # allocator, or shared-storage effects.
            runner._kimi_k3_fhmoe_mxfp4_enabled = False
            post_fusion_baseline_actual = run_forward(
                layer,
                vllm_config,
                inputs[0],
            )
            post_fusion_baseline_us = measure(
                layer,
                vllm_config,
                inputs,
                args.warmup,
                args.iterations,
                args.samples,
            )

            optimized_error = torch.tensor(
                relative_rmse(optimized_actual, reference),
                dtype=torch.float64,
                device=device,
            )
            post_fusion_baseline_error = torch.tensor(
                relative_rmse(post_fusion_baseline_actual, reference),
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(optimized_error, op=dist.ReduceOp.MAX)
            dist.all_reduce(
                post_fusion_baseline_error,
                op=dist.ReduceOp.MAX,
            )

            if rank == 0:
                baseline_median = statistics.median(baseline_us)
                optimized_median = statistics.median(optimized_us)
                post_fusion_baseline_median = statistics.median(post_fusion_baseline_us)
                print(f"Kimi-K3 M={args.batch_size} MXFP4 TP8 single-layer forward")
                print(f"experts: {args.num_experts}, top-k: {TOPK}")
                print(f"separate baseline: {baseline_median:.3f} us")
                print(f"unified FHMoE:     {optimized_median:.3f} us")
                print(f"speedup:           {baseline_median / optimized_median:.3f}x")
                print(f"max RRMSE:         {float(optimized_error):.6f}")
                print(f"post-init baseline: {post_fusion_baseline_median:.3f} us")
                print(f"post-init RRMSE:    {float(post_fusion_baseline_error):.6f}")
                print(f"baseline samples:  {baseline_us}")
                print(f"FHMoE samples:     {optimized_us}")
                print(f"post-init samples: {post_fusion_baseline_us}")
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> None:
    args = parse_args()
    if (
        min(
            args.num_experts,
            args.batch_size,
            args.rotations,
            args.warmup,
            args.iterations,
            args.samples,
        )
        <= 0
    ):
        raise ValueError("benchmark arguments must be positive")
    if args.num_experts < TOPK:
        raise ValueError(f"num-experts must be at least {TOPK}")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch-size must be in [1, 32]")

    spawn(
        worker,
        args=(8, get_open_port(), args),
        nprocs=8,
        join=True,
    )


if __name__ == "__main__":
    main()
