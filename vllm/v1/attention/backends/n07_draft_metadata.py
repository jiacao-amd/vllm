"""Guarded current-nightly ROCm draft metadata alias contract; no GPU math."""

from __future__ import annotations

import os

TENSOR_FIELDS = {
    "query_start_loc": "query_start_loc",
    "seq_lens": "seq_lens",
    "block_table": "block_table_tensor",
    "slot_mapping": "slot_mapping",
}
FIELDS = frozenset((
    "num_actual_tokens", "max_query_len", "query_start_loc", "max_seq_len",
    "seq_lens", "block_table", "slot_mapping", "scheduler_metadata", "causal",
))


def eligible(config) -> bool:
    """Opt in only to the fixed TP4 MiniMax Eagle3 lane; keep all other defaults."""
    if os.getenv("MM3_N07_FUSED_DRAFT") != "1":
        return False
    model, parallel, spec = (
        config.model_config, config.parallel_config, config.speculative_config,
    )
    return (
        spec is not None and model.hf_config.model_type == "minimax_m3_vl"
        and str(model.dtype) == "torch.bfloat16"
        and model.get_hidden_size() == 6144
        and spec.method == "eagle3" and spec.num_speculative_tokens == 3
        and not spec.parallel_drafting and not spec.enable_adaptive_verification
        and not spec.uses_dynamic_speculative_decoding()
        and parallel.tensor_parallel_size == 4 and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1 and parallel.decode_context_parallel_size == 1
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and config.lora_config is None
        and getattr(config, "kv_transfer_config", None) is None
    )


def tensor_signature(tensor):
    """Only host-visible metadata; no item/cpu/copy/allocation during capture."""
    return (
        tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
        str(tensor.dtype), str(tensor.device),
    )


def certify(metadata, common):
    """Prove all device fields are original live views and no derived cache exists."""
    if frozenset(metadata.__dataclass_fields__) != FIELDS:
        raise ValueError("ROCm metadata fields changed; re-audit all derived fields")
    if metadata.scheduler_metadata is not None or metadata.causal is not True:
        raise ValueError("N07 supports only causal metadata without scheduler caches")
    if metadata.max_query_len != 1:
        raise ValueError("N07 refresh is for one-token uniform draft decode only")
    if metadata.max_seq_len != common.max_seq_len or metadata.max_seq_len < 1:
        raise ValueError("The native whole-draft-horizon scalar bound must be retained")
    for target, source in TENSOR_FIELDS.items():
        if getattr(metadata, target) is not getattr(common, source):
            raise ValueError(f"Metadata stopped aliasing its live common buffer: {target}")
    rows = metadata.seq_lens.shape[0]
    if (metadata.query_start_loc.shape != (rows + 1,)
            or metadata.block_table.ndim != 2 or metadata.block_table.shape[0] != rows
            or metadata.slot_mapping.ndim != 1
            or rows <= 0 or metadata.num_actual_tokens <= 0
            or metadata.slot_mapping.shape[0] < metadata.num_actual_tokens):
        raise ValueError("Unsupported uniform-decode metadata geometry")
    expected = {
        "query_start_loc": "torch.int32", "seq_lens": "torch.int32",
        "block_table": "torch.int32", "slot_mapping": "torch.int64",
    }
    if any(str(getattr(metadata, name).dtype) != dtype for name, dtype in expected.items()):
        raise ValueError("Native metadata integer precision changed")
    devices = {str(getattr(metadata, name).device) for name in TENSOR_FIELDS}
    if len(devices) != 1 or not next(iter(devices)).startswith("cuda"):
        raise ValueError("All live metadata views must share one GPU")
    metadata._n07_live_contract = {
        "tensors": {name: tensor_signature(getattr(metadata, name)) for name in TENSOR_FIELDS},
        "scalars": (metadata.num_actual_tokens, metadata.max_query_len,
                    metadata.max_seq_len, metadata.causal),
    }
    return metadata


def refresh(metadata):
    """Validated no-op: native slot/state kernels already update the aliased buffers.

    This backend has no scheduler/derived device metadata. The unchanged native
    fused loop computes slot mappings before each step and advances positions,
    sampling keys, sequence lengths and hidden states through its original
    kernels. A scalar bound covers the whole proposal, not merely the first step.
    """
    contract = getattr(metadata, "_n07_live_contract", None)
    if contract is None or metadata.scheduler_metadata is not None:
        raise ValueError("N07 update requires metadata certified at build time")
    if frozenset(metadata.__dataclass_fields__) != FIELDS:
        raise ValueError("Unknown derived metadata field")
    actual = {name: tensor_signature(getattr(metadata, name)) for name in TENSOR_FIELDS}
    scalars = (metadata.num_actual_tokens, metadata.max_query_len,
               metadata.max_seq_len, metadata.causal)
    if actual != contract["tensors"] or scalars != contract["scalars"]:
        raise ValueError("Live metadata addresses/layout or native bounds changed")
