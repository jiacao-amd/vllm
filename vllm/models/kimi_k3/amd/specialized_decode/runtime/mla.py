# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the conc1 MLA specialization when Kimi's mono executor is built.

Destination in PR 60683: specialized_decode/runtime/mla.py. The kernel module
is specialized_decode/mono/mla_decode.py. Each attention layer owns its output;
serial layers share one 31.75 MiB scratch workspace per device. The existing
mono runtime refuses DBO, so attention launches on a rank are ordered.
"""

from typing import Any

import torch

from ..mono.mla_decode import KimiMLADecodePlan

_workspaces: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def install_on_layer(layer: Any) -> None:
    wrapper = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
    impl = getattr(getattr(wrapper, "mla_attn", None), "impl", None)
    if impl is None:
        return
    if getattr(impl, "_kimi_flydsl_mla_plans", None) is not None:
        return
    if (
        getattr(impl, "dcp_world_size", None) != 8
        or getattr(impl, "_decode_num_heads", None) != 96
        or getattr(impl, "scale", None) != 192**-0.5
    ):
        return
    original = impl._forward_mqa
    plans: dict[tuple[torch.device, int, int, int], KimiMLADecodePlan] = {}
    impl._kimi_flydsl_mla_plans = plans

    def forward(q, kv_cache, metadata, attention_layer):
        decode = getattr(metadata, "decode", None)  # type: Any
        eligible = (
            decode is not None
            and metadata.causal
            and metadata.num_actual_tokens == metadata.num_decode_tokens == 8
            and decode.max_qo_len == 8
            and decode.attn_out_dtype == torch.bfloat16
            and getattr(decode.dcp_route, "name", None) == "CPRR"
            and decode.cp_world_size == 8
            and decode.g_kv_indptr is not None
            and decode.g_kv_indptr.numel() == 2
            and decode.qo_indptr.numel() == 2
            and decode.paged_kv_indices is not None
            and decode.paged_kv_indices.numel() > 0
            and decode.paged_kv_indices.dtype == torch.int32
            and decode.paged_kv_indices.is_contiguous()
            and kv_cache.dtype == torch.float8_e4m3fn
            and kv_cache.shape[-1] == 576
            and kv_cache.is_contiguous()
        )
        if not eligible:
            return original(q, kv_cache, metadata, attention_layer)
        if isinstance(q, tuple):
            if (
                len(q) != 2
                or any(t.dtype != torch.float8_e4m3fn for t in q)
                or tuple(q[0].shape) != (8, 96, 512)
                or tuple(q[1].shape) != (8, 96, 64)
            ):
                return original(q, kv_cache, metadata, attention_layer)
            packed_q = torch.cat(q, dim=-1)
        else:
            packed_q = q
        if (
            packed_q.dtype != torch.float8_e4m3fn
            or tuple(packed_q.shape) != (8, 96, 576)
            or not packed_q.is_contiguous()
        ):
            return original(q, kv_cache, metadata, attention_layer)
        kv = kv_cache.view(-1, 576)
        indices = decode.paged_kv_indices
        key = (packed_q.device, kv.shape[0], indices.numel(), decode.cp_rank)
        plan = plans.get(key)
        if plan is None:
            plan = KimiMLADecodePlan(
                max_local_tokens=indices.numel(),
                cache_slots=kv.shape[0],
                cp_rank=decode.cp_rank,
                device=packed_q.device,
                workspace=_workspaces.get(packed_q.device),
            )
            _workspaces[packed_q.device] = plan.partial, plan.stats
            plans[key] = plan
        return plan(
            packed_q,
            kv,
            indices,
            decode.g_kv_indptr[1:2],
            attention_layer._q_scale,
            attention_layer._k_scale,
        )

    impl._forward_mqa = forward
