"""One fixed local protocol for official N0 and cumulative nightly candidates."""

BASE_IMAGE = (
    "vllm/vllm-openai-rocm@sha256:"
    "c6cb4ab021e31691f63a5757278153fbe728a71c722651a4549ddbb5ac3067ab"
)
MAIN = "/models/main/snapshots/229447a2d64149ffb45e454ebeb74b7a9a45ce7d"
DRAFT = "/models/draft/snapshots/96692486b5fd38ebf8fd2a5f6bb53427d30819a8"
SERVED = "amd/MiniMax-M3-MXFP4"
COMPILATION = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        20, 22, 24, 26, 28, 30, 32, 34, 36, 48, 64, 72, 80, 88, 96, 104,
        112, 120, 128,
    ],
    "max_cudagraph_capture_size": 128,
}
SPECULATIVE = {
    "method": "eagle3", "model": DRAFT, "num_speculative_tokens": 3,
    "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
    "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.78,
}
ENGINE = {
    "model": MAIN, "tokenizer": MAIN, "trust_remote_code": True,
    "tensor_parallel_size": 4, "block_size": 128,
    "gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8",
    "enable_prefix_caching": True, "enable_chunked_prefill": True,
    "language_model_only": True, "max_num_batched_tokens": 32768,
    "max_num_seqs": 48, "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
    "moe_backend": "aiter", "attention_config": {"indexer_kv_dtype": "fp8"},
    "compilation_config": COMPILATION, "speculative_config": SPECULATIVE,
    "seed": 0,
}
ENVIRONMENT = {
    "HIP_VISIBLE_DEVICES": "4,5,6,7", "OMP_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    "HOME": "/cache/home", "USER": "jiacao", "LOGNAME": "jiacao",
    "PYTHONPATH": "/suite",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "0",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": "256",
    "VLLM_CACHE_ROOT": "/cache/vllm", "TRITON_CACHE_DIR": "/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
    "FLYDSL_RUNTIME_CACHE_DIR": "/cache/flydsl",
}
# These are native engine dummy forwards, not HTTP/generate requests.
# Every variant receives the same compiler-coverage workload.
COMPILE_CASES = [
    {"tokens": 96, "uniform_decode": True, "context_len": 8192},
    {"tokens": 96, "uniform_decode": True, "context_len": 9216},
    *[
        {"tokens": tokens, "uniform_decode": False, "context_len": 0}
        for tokens in (1, 24, 96, 128, 192, 256, 260, 288, 320, 512,
                       1024, 2048, 4096, 8192, 16384, 32768)
    ],
]
PROTOCOL_ID = "official-nightly-tp4-c24-8k1k-http0-dummy-compile-v1"


def server_arguments(port=6101):
    import json
    values = [
        "serve", MAIN, "--served-model-name", SERVED, "--host", "0.0.0.0",
        "--port", str(port), "--trust-remote-code", "--tensor-parallel-size", "4",
        "--block-size", "128", "--gpu-memory-utilization", "0.9",
        "--kv-cache-dtype", "fp8", "--enable-prefix-caching",
        "--enable-chunked-prefill", "--language-model-only",
        "--max-num-batched-tokens", "32768", "--max-num-seqs", "48",
        "--stream-interval", "20", "--attention-backend", "ROCM_AITER_UNIFIED_ATTN",
        "--moe-backend", "aiter", "--attention-config", '{"indexer_kv_dtype":"fp8"}',
        "--compilation-config", json.dumps(COMPILATION),
        "--speculative-config", json.dumps(SPECULATIVE),
        "--tool-call-parser", "minimax_m3", "--enable-auto-tool-choice",
        "--reasoning-parser", "minimax_m3",
        "--default-chat-template-kwargs", '{"thinking_mode":"enabled"}',
    ]
    return values
