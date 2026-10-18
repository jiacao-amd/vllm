"""R7: A48 TN64 G1 only, feeding the unchanged A0 CK-Tile Stage2 layout."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
from pathlib import Path

_active = ContextVar("r07_g1_call", default=None)
_state = None


@dataclass
class Call:
    enabled: bool
    launches: int = 0


def configure_for_layer(layer, config, *, use_aiter_fse):
    """Scope the Stage1 replacement inside the existing opaque MoE call."""
    import torch
    import aiter.fused_moe as fm
    from aiter.ops.flydsl.kernels import moe_2stage_a16wmix as port
    from vllm.logger import init_logger
    from .fixed_g1 import compile_fixed

    if getattr(layer, "_r07_g1_enabled", False):
        return
    moe = layer.moe_config
    supported = (
        use_aiter_fse and config.hidden_size == 6144
        and config.intermediate_size == 3072 and config.num_local_experts == 128
        and config.num_experts_per_tok == 4 and config.n_shared_experts == 1
        and config.routed_scaling_factor == 2 and config.swiglu_limit == 7
        and moe.tp_size == 4 and moe.ep_size == 1 and moe.dp_size == 1
        and not moe.is_sequence_parallel and not layer.enable_dbo
        and layer.routed_input_transform is None and layer.routed_output_transform is None
    )
    if not supported:
        raise RuntimeError("R7 requires the fixed A0 MiniMax TP4/FSE configuration")
    global _state
    if _state is None:
        from aiter.ops.flydsl.kernels import act as global_act
        from . import act as private_act
        for module, expected in (
            (global_act, "0602318cdc3ee64fcef18ad79d731848f26f479c7a2badb65046e7727e10ddee"),
            (private_act, "1632853f42132b68e082c5dea073bd6afaf2d00b9b87ec32952a0d75ca2a96bd"),
        ):
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
                raise RuntimeError("R7 requires unchanged A0/global and exact A48/private activation helpers")
        original = fm.cktile_moe_stage1
        signature = inspect.signature(original)

        @wraps(original)
        def stage1(*args, **kwargs):
            call = _active.get()
            if call is None or not call.enabled:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            b = bound.arguments
            a, w1, w2, out = (b[name] for name in ("hidden_states", "w1", "w2", "out"))
            if not (
                tuple(a.shape) == (96, 6144) and a.dtype == torch.bfloat16 and a.is_contiguous()
                and tuple(w1.shape) == (129, 1536, 3072)
                and tuple(w2.shape) == (129, 6144, 384)
                and w1.dtype == fm.dtypes.fp4x2 and w1.is_contiguous()
                and out is not None and tuple(out.shape) == (96, 5, 768)
                and out.dtype == torch.bfloat16 and out.is_contiguous()
                and b["block_m"] == 16 and b["topk"] == 5
                and b["activation"] == fm.ActivationType.Swiglu
                and b["dtype"] == torch.bfloat16 and b["a1_scale"] is None
                and b["w1_scale"] is not None and b["sorted_weights"] is None
                and b["n_pad_zeros"] == b["k_pad_zeros"] == 0 and b["bias1"] is None
                and b["post_activation_layout"] == "standard" and b["split_k"] >= 2
            ):
                raise RuntimeError("R7 A0 CK-Tile/BM16/slot-layout Stage1 contract changed")
            eids = b["sorted_expert_ids"]
            tids = b["sorted_token_ids"].to(torch.int32).contiguous()
            counts = b["num_valid_ids"].to(torch.int32).contiguous()
            w1_bytes = w1.view(torch.uint8)
            scales = b["w1_scale"].view(torch.uint8).contiguous().view(-1)
            if eids.dtype != torch.int32 or not eids.is_contiguous():
                raise RuntimeError("R7 requires native contiguous int32 expert metadata")
            port._run_compiled(
                compile_fixed(), a.data_ptr(), w1_bytes.data_ptr(), scales.data_ptr(),
                eids.data_ptr(), counts.data_ptr(), tids.data_ptr(), 96, eids.numel() * 12,
                1.0, 1.0, 1.0, 1.0, 7.0, out.data_ptr(), torch.cuda.current_stream(),
            )
            call.launches += 1
            return out

        fm.cktile_moe_stage1 = stage1
        fm.get_2stage_cfgs.cache_clear()
        _state = (original, stage1)
        init_logger(__name__).info("R7 enabled: TN64 G1, BM16 sort, A0 slot-layout CK-Tile G2/TP retained")
    original_forward = layer._forward_impl

    @wraps(original_forward)
    def forward(hidden_states, *args, **kwargs):
        call = Call(tuple(hidden_states.shape) == (96, 6144) and hidden_states.dtype == torch.bfloat16)
        token = _active.set(call)
        try:
            result = original_forward(hidden_states, *args, **kwargs)
            if call.enabled and call.launches != 1:
                raise RuntimeError("R7 requires exactly one TN64 G1 launch in each target MoE call")
            return result
        finally:
            _active.reset(token)

    layer._forward_impl = forward
    layer._r07_g1_enabled = True
