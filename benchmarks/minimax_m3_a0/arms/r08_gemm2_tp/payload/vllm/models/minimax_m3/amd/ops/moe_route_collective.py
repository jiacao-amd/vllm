# SPDX-License-Identifier: Apache-2.0
"""Private A37 native-atomic M96 TP4 experiment."""
import sys
import os
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import torch
from vllm.logger import init_logger
logger = init_logger(__name__)
_state = None
_original_stage2 = None
_EXPECTED_STAGE2 = 'flydsl_moe2_abf16_wfp4_bf16_t16x256x256_atomic_bnt2_persist'

@dataclass
class _Call:
    enabled: bool
    launches: int = 0
_active_call: ContextVar[_Call | None] = ContextVar('mm3_route_call', default=None)

def _reduce_native(result):
    from vllm.distributed import tensor_model_parallel_all_reduce
    if not isinstance(result, torch.Tensor):
        raise RuntimeError('MiniMax FSE route collective expects one tensor output')
    return tensor_model_parallel_all_reduce(result)

def _wrap_forward(original):
    """Always return a reduced tensor from the opaque MoE call."""

    @wraps(original)
    def forward(hidden_states, *args, **kwargs):
        call = _Call(enabled=hidden_states.shape == (96, 6144) and hidden_states.dtype == torch.bfloat16)
        token = _active_call.set(call)
        try:
            result = original(hidden_states, *args, **kwargs)
            if call.enabled and call.launches != 1:
                raise RuntimeError('MiniMax M96 requires exactly one fused route collective launch')
            return result if call.enabled else _reduce_native(result)
        finally:
            _active_call.reset(token)
    return forward

class _RouteState:

    def __new__(cls, group):
        from ._moe_atomic_pipeline.atomic_state import AtomicPipelineState
        AtomicPipelineState.column_first = True
        AtomicPipelineState.clear_in_sort = True
        result = AtomicPipelineState(group)
        result.group = group
        return result

def _stage2(inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk, **kwargs):
    call = _active_call.get()
    if call is None or not call.enabled or out.shape[0] != 96:
        return _original_stage2(inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk, **kwargs)
    from aiter import ActivationType, dtypes
    valid = _state is not None and tuple(out.shape) == (96, 6144) and (out.dtype == torch.bfloat16) and out.is_contiguous() and (tuple(inter_states.shape) == (96, 5, 768)) and (inter_states.dtype == torch.bfloat16) and inter_states.is_contiguous() and (tuple(w2.shape) == (129, 6144, 384)) and (w2.dtype == dtypes.fp4x2) and (topk == 5) and (kwargs.get('block_m') == 16) and (kwargs.get('sorted_weights') is not None) and (kwargs.get('w2_scale') is not None) and (kwargs.get('a2_scale') is None) and (kwargs.get('bias2') is None) and (kwargs.get('n_pad_zeros', 0) == kwargs.get('k_pad_zeros', 0) == 0) and (kwargs.get('activation') == ActivationType.Swiglu) and (kwargs.get('kernel_name', '') == '') and (sorted_token_ids.dtype == torch.int32) and sorted_token_ids.is_contiguous() and (sorted_expert_ids.dtype == torch.int32) and sorted_expert_ids.is_contiguous()
    if not valid:
        raise RuntimeError('R8 requires unchanged A0 CK-Tile G1 slot output and BM16 metadata')
    result = _state.stage2(inter_states.view(96 * 5, 768), w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, kwargs['w2_scale'], kwargs['sorted_weights'])
    call.launches += 1
    return result

def configure_for_layer(layer, config, *, use_aiter_fse: bool) -> bool:
    """Enable only the fixed M3 TP4 FSE contract, before AITER metadata caching."""
    if os.getenv('MM3_MOE_ROUTE_COLLECTIVE_EXPERIMENT', '0') != '1':
        return False
    if getattr(layer, '_mm3_route_collective_enabled', False):
        return True
    from vllm.distributed import get_tp_group
    from vllm.platforms.rocm import on_gfx950
    moe = layer.moe_config
    supported = on_gfx950() and use_aiter_fse and (config.hidden_size == 6144) and (config.intermediate_size == 3072) and (config.num_local_experts == 128) and (config.num_experts_per_tok == 4) and (config.n_shared_experts == 1) and (config.routed_scaling_factor == 2) and (config.scoring_func == 'sigmoid') and (config.swiglu_limit == 7) and (moe.tp_size == 4) and (moe.ep_size == 1) and (moe.dp_size == 1) and (not moe.is_sequence_parallel) and (not moe.skip_final_all_reduce) and (not layer.enable_dbo) and (layer.routed_input_transform is None) and (layer.routed_output_transform is None) and (layer.routed_scaling_factor == 1.0) and (layer._shared_experts is None) and (not layer._fused_output_is_reduced)
    if not supported:
        raise RuntimeError('Unsupported MiniMax-M3 route collective configuration')
    global _state, _original_stage2
    group = get_tp_group().cpu_group
    if _state is None:
        import aiter.fused_moe as fm
        _state = _RouteState(group)
        from ._moe_atomic_pipeline.sort_clear import install
        install(sys.modules[__name__])
        _original_stage2 = fm.cktile_moe_stage2
        fm.cktile_moe_stage2 = _stage2
        fm.get_2stage_cfgs.cache_clear()
    elif _state.group is not group:
        raise RuntimeError('A route collective workspace cannot span TP groups')
    layer._forward_impl = _wrap_forward(layer._forward_impl)
    layer._mm3_route_collective_enabled = True
    return True
