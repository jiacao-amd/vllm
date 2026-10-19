"""N16 TK128 replacement in the ancestral N14 helper; same opaque MoE op."""

from dataclasses import replace
from functools import lru_cache, partial
import importlib
import inspect
import json
import os

NATIVE_G1 = "flydsl_moe1_abf16_wfp4_bf16_t16x32x128_w4_xcd1_kw2"
SELECTED_G1 = "flydsl_moe1_abf16_wfp4_bf16_t16x64x128_w2_xcd1_kw2"
NATIVE_G2 = "flydsl_moe2_abf16_wfp4_bf16_t16x256x128_atomic_bnt2_persist"
_REPORTED = False


def _tensor(value, shape, dtype, device):
    """Metadata only: no device-value read, tensor copy or synchronization."""
    return (
        value is not None and tuple(value.shape) == shape
        and str(value.dtype) == dtype and value.device == device
        and value.is_contiguous()
    )


def eligible(v):
    """Keep every unsupported shape/layout/feature on the original public path."""
    hidden = v["hidden_states"]
    device = hidden.device
    if getattr(device, "type", None) != "cuda":
        return False
    if not _tensor(hidden, (96, 6144), "torch.bfloat16", device):
        return False
    required = (
        ("w1", (129, 1536, 3072), "torch.float4_e2m1fn_x2"),
        ("w2", (129, 6144, 384), "torch.float4_e2m1fn_x2"),
        ("w1_scale", (129, 1536, 192), "torch.uint8"),
        ("w2_scale", (129, 6144, 24), "torch.uint8"),
        ("topk_weight", (96, 5), "torch.float32"),
        ("topk_ids", (96, 5), "torch.int32"),
    )
    if not all(_tensor(v[key], shape, dtype, device) for key, shape, dtype in required):
        return False
    if not all(getattr(v[key], "is_shuffled", False) for key in ("w1", "w2")):
        return False
    if any(v[key] is not None for key in (
        "expert_mask", "a1_scale", "a2_scale", "num_local_tokens", "bias1", "bias2",
        "shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale",
        "stage2_scatter", "output", "quant_type_a", "quant_dtype_a", "quant_dtype_a2",
    )):
        return False
    return (
        getattr(v["activation"], "name", None) == "Swiglu"
        and getattr(v["quant_type"], "name", None) == "per_1x32"
        and getattr(v["gate_mode"], "value", v["gate_mode"]) == "separated"
        and not v["doweight_stage1"] and v["hidden_pad"] == v["intermediate_pad"] == 0
        and v["shared_expert_id"] == -1 and v["splitk"] == 0
        and v["block_size_M"] in (None, 0, -1) and v["swiglu_limit"] == 7.0
        and v["dtype"] in (None, hidden.dtype)
    )


@lru_cache(maxsize=1)
def _native():
    """Resolve the pinned module explicitly, not AITER's same-named reexport."""
    return importlib.import_module("aiter.fused_moe")


def platform_supported(fm):
    """Use the native runtime's actual architecture/CU and TP checks."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    return (fm.get_gfx_runtime() == "gfx950" and fm.get_cu_num() == 256
            and get_tensor_model_parallel_world_size() == 4)


def transform(metadata):
    """Pure metadata copy: only the exact native stage-1 kernelName may change."""
    if (
        metadata.block_m != 16 or metadata.ksplit != 0 or metadata.run_1stage
        or metadata.full_impl is not None or metadata.fuse_quant or metadata.skip_inter_quant
        or not isinstance(metadata.stage1, partial) or not isinstance(metadata.stage2, partial)
        or metadata.stage1.keywords.get("kernelName") != NATIVE_G1
        or metadata.stage2.keywords.get("kernelName") != NATIVE_G2
    ):
        return metadata
    fm = _native()
    if (metadata.stage1.func is not fm._flydsl_stage1_wrapper
            or metadata.stage2.func is not fm._flydsl_stage2_wrapper):
        return metadata
    stage1 = partial(metadata.stage1.func, *metadata.stage1.args,
                     **{**metadata.stage1.keywords, "kernelName": SELECTED_G1})
    result = replace(metadata, stage1=stage1)
    global _REPORTED
    if os.getenv("MM3_N16_AUDIT") == "1" and not _REPORTED:
        print("MM3_N16_G1_SELECTED " + json.dumps({
            "M": 96, "H": 6144, "I": 768, "E": 129, "topk": 5,
            "BM": 16, "TN": 64, "TK": 128, "kw": 2, "waves_per_eu": 2,
            "stage1": SELECTED_G1, "stage2": NATIVE_G2,
            "selection_only_not_gpu_execution_proof": True,
        }), flush=True)
        _REPORTED = True
    return result


@lru_cache(maxsize=1)
def _signature(native):
    return inspect.signature(native)


def private_kwargs(values):
    """Mirror public fused_moe -> fused_moe_ conversion on the guarded no-scatter path."""
    result = dict(values)
    result.pop("splitk")  # Public fused_moe does not forward it; guard requires zero.
    for name in ("shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale", "shared_expert_id"):
        result.pop(name)  # The guarded path cannot enter public _fhmoe dispatch.
    result["activation"] = values["activation"].value
    result["quant_type"] = values["quant_type"].value
    result["block_size_M"] = values["block_size_M"] or -1
    result["_metadata_transform"] = transform
    return result


def fused_moe(native, *args, **kwargs):
    """Retain the public function verbatim for all noneligible calls."""
    if os.getenv("MM3_N16_G1_TN64_K128") != "1":
        return native(*args, **kwargs)
    bound = _signature(native).bind(*args, **kwargs)
    bound.apply_defaults()
    if not eligible(bound.arguments):
        return native(*args, **kwargs)
    fm = _native()
    if native is not fm.fused_moe or not platform_supported(fm):
        return native(*args, **kwargs)
    return fm._fused_moe_impl(**private_kwargs(bound.arguments))
