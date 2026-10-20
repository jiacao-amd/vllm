# Kimi-K3 conc1 MLA on MI355X

This branch adds a FlyDSL MLA decode kernel to the AMD specialized decode
runtime from [vLLM PR #60683](https://github.com/vllm-project/vllm/pull/60683).
It targets conc1, seven speculative tokens, TP8/DCP8 and FP8 KV cache.
Single-rank O+LSE measurements outperform AITER ASM at the tested 8k, 64k and
256k lengths. Full K3 engine integration, TP8 TPOT and model quality evaluation
have **not** been run.

## Implementation

- The kernel is in
  `vllm/models/kimi_k3/amd/specialized_decode/mono/mla_decode.py`.
  Its `KimiMLADecodePlan` owns preallocated output and split scratch buffers.
- `specialized_decode/runtime/mla.py` installs the specialization on the
  attention backend instance when `FfnLayer` is constructed. It selects only
  one-request, eight-token, causal DCP8 CPRR decode with 96 gathered heads,
  E4M3FN Q/KV and BF16 output. Other inputs use the original backend.
- MLA still launches stage1 and merge before the existing DCP merge, V
  projection, gate and K2. It is not fused into K2's persistent launch.
- Layers share 31.75 MiB of scratch per device and own separate O/LSE buffers.
  Calls sharing scratch must be ordered, as they are in the mono runtime.

The input is Q `[8,96,576]` and token-indexed KV `[physical_slots,576]`.
The first 512 KV channels are also V. All 576 QK channels contribute, including
the final 64 channels when rotary embedding is disabled. Output is BF16
`[8,96,512]`; natural-log LSE is FP32 `[8,96]`.
The softmax scale is `192**-0.5`, with device FP32 Q/KV dequantization scales.
Scalar scales and one-element vectors are both accepted.

Global sequence length includes the eight verify tokens. For query `t`, rank
`r` sees `max(0, (global_len - 8 + t + 8 - r) // 8)` local tokens.
The index vector maps these logical local tokens to physical cache slots.
LSE is needed by the existing DCP8 merge, including the AITER path; there is no
external max-score output. The standalone plan supports `return_lse=False`.

## Optimizations

The MFMA32 layout, online softmax and PV rearrangement originate from
[PyTorch PR #194309](https://github.com/pytorch/pytorch/pull/194309).
The MLA specialization changes the following:

1. Native gfx950 FP8 `32x32x64_f8f6f4` MFMA for QK and PV. It computes the
   actual 96 heads, avoiding AITER's 96-to-128 head padding.
2. A 64-token KV tile with K/V sharing one LDS layout; Q remains in registers.
   Double buffering prefetches KV while the current tile is processed.
3. Each four-wave CTA computes all 512 output dimensions, reusing QK and
   softmax across the full latent output.
4. Per-chunk accumulator rescaling and scheduling barriers control register
   lifetimes. The measured ISA uses 72 KiB LDS and no private scratch segment.
   LLVM reports four VGPR spills, so this is not a claim of zero compiler spills.
5. Up to 42 active splits, with device-side selection and 16/32/42-way merge
   paths. Sequence length changes within the same captured graph need no host
   scalar read or graph recapture.
6. FP32 accumulation and merge, BF16 split output, and FP8 probabilities scaled
   by 256 before conversion. This reduces split traffic while controlling
   probability quantization error.

These are the changes in the winning configuration; the timing below measures
their combined effect, not an attribution of speedup to each optimization.

## Recorded performance

Single MI355X, rank 0, seed 1234, random physical 64-token pages. Values are
microseconds for complete O+LSE graph calls. The best ASM median is selected
from split caps 64/128/256/512; the PR's deployed CPRR cap is 256.

| Global KV tokens, including verify | FlyDSL | Best ASM | Speedup |
| ---: | ---: | ---: | ---: |
| 8192 | 20.008 | 26.156 | 1.307x |
| 8200 | 24.155 | 28.328 | 1.173x |
| 9216 | 24.691 | 28.782 | 1.166x |
| 65536 | 32.247 | 43.974 | 1.364x |
| 65544 | 32.229 | 44.501 | 1.381x |
| 262144 | 61.512 | 69.218 | 1.125x |

An 8192-token prompt's first eight-token verify step corresponds to **8200**.
Independent rank 7 / seed 2026 confirmation at 8200/65544/262152 tokens measured
1.152x/1.427x/1.178x against the best ASM cap from 64/128/256.
Random token-level physical permutations were also measured. See
`results/comparison_final.json` and `results/comparison_confirm.json` for every
sample and the precise configuration.

Each graph captures 40 complete calls; each timing replays it 20 times. Nine
rounds rotate and reverse implementation order, with medians reported. Outputs
are poisoned after capture and checked after replay to reject empty graphs.
Both paths use the same data, scales, causal mask and required outputs. AITER
head padding and metadata construction are excluded from timing. These are
warm GPU graph timings, excluding compilation, allocation and host overhead.

## Validation and provenance

`results/validation_final.json` contains 111 passing records: all eight ranks,
dynamic lengths from 8 through 262144, split thresholds, page mappings, NaN
padding, nonunit scales, multiple seeds and a simulated eight-rank DCP merge
against full FP32 attention. O-only output was also checked. The GPU adapter
check passed dispatch, tuple Q, scalar scales, fallback, idempotence, shared
scratch, separate outputs and changing length in one graph.

At Q/K standard deviation 0.5, error is close to ASM. At standard deviation 1.5,
the maximum absolute error matched ASM on three tested seeds, but RMS error was
about 10-13% higher. Model-level accuracy remains unvalidated. Typical LSE
maximum error is approximately `2e-6`; the check uses `atol=2e-4, rtol=2e-5`.

Recorded environment:

- Image: `vllm/vllm-openai-rocm@sha256:520e098f54ec96837228aaddea0a833cbf6374dbdffd480c1d65f26ffab7701b`.
- Torch `2.12.0+rocm10.0.0`, HIP `7.15.26333`, FlyDSL `0.3.4.1`, AITER from that image.
- MI355X gfx950, 256 CUs.
- Tested kernel SHA256: `864f7488249f8df6778b3f8e786c9ce1a6f14705a039b2d3e6f0c257070bae22`.
- Original integration source: PR #60683 head `033f35bf2f26497c882b829a583b0e3eb001f2a3`.
- This branch's base: PR #60683 head `989829c0beb9d5270fbf4c560565c0212b7d0d88`.
  The relevant runtime layer and AITER backend files are identical at both heads.
- FlexAttention source head: `5bc1e0fd3e4e89f762a5b6cc805814ff8b64f156`.

The stored GPU results precede this vLLM packaging. The kernel has been formatted
for vLLM and uses `torch.accelerator.current_device_index()` for default device
selection. Its GPU builder AST is unchanged. Scripts retain the production
benchmark and validation paths, with experimental tuning branches removed.
These packaged scripts have not been rerun on GPU. AI assistance was used for
the implementation, validation harness and branch preparation.
`results/provenance.json` records the packaged source hash and the hashes of
the unchanged GPU result files. Branch preparation checks include Python
compilation, JSON parsing, GPU builder AST equivalence and all pre-commit hooks.

## Reproduction

Use an allocated MI355X with the image above and this checkout available.
The image supplies Torch, AITER and FlyDSL. Create a virtual environment that
can see these packages; use the repository's documented editable-install
workflow when running a complete vLLM engine.

From the repository root:

```bash
uv venv --python 3.12 --system-site-packages .venv
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python benchmarks/kernels/kimi_mla/validate.py \
  --output /tmp/kimi-validation.json
.venv/bin/python benchmarks/kernels/kimi_mla/adapter_check.py
.venv/bin/python benchmarks/kernels/kimi_mla/compare.py \
  --config benchmarks/kernels/kimi_mla/production_config.json \
  --asm-splits 64 128 256 512 --output /tmp/kimi-comparison.json
.venv/bin/python benchmarks/kernels/kimi_mla/compare.py \
  --config benchmarks/kernels/kimi_mla/confirm_config.json \
  --rank 7 --seed 2026 --asm-splits 64 128 256 \
  --output /tmp/kimi-confirmation.json
```

For serving, the adapter is installed when the existing
`additional_config={"decode_backend": "latency_specialized"}` path constructs
its MLA layers. Full engine and quality evaluation are still required before
treating the measured operator speedups as serving improvements.
