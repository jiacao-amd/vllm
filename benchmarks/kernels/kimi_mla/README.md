# Kimi-K3 conc1 MLA on MI355X

A standalone three-file bundle for conc1, seven speculative tokens, TP8/DCP8
and FP8 KV cache:

- [mla_decode.py](mla_decode.py): FlyDSL kernel and `KimiMLADecodePlan` API.
- [benchmark.py](benchmark.py): input generation, FP32 reference, AITER ASM
  baseline and alternating complete O+LSE graph timings.
- [README.md](README.md): shapes, optimizations, results and usage.

## Kernel API

Q is FP8 E4M3FN `[8,96,576]`, with 96 heads after DCP gather. KV is FP8
E4M3FN `[physical_slots,576]`; the first 512 channels are also V. All 576 QK
channels contribute, including the final 64 channels without rotary embedding.
Output is BF16 `[8,96,512]`, with natural-log FP32 LSE `[8,96]`.
The softmax scale is `192**-0.5`, with device FP32 Q/KV dequantization scales.

The contiguous int32 index vector maps logical rank-local tokens to physical
cache slots. `global_seq_len` is device int32 `[1]`, including eight verify
tokens. Query `t` on rank `r` sees
`max(0, (global_len - 8 + t + 8 - r) // 8)` local tokens. The caller must keep
length within the allocated index capacity and provide valid slot indices.

```python
from mla_decode import KimiMLADecodePlan

# Allocate and warm up before graph capture.
plan = KimiMLADecodePlan(
    max_local_tokens=indices.numel(),
    cache_slots=kv.shape[0],
    cp_rank=rank,
    device=q.device,
)
out, lse = plan(q, kv, indices, global_seq_len, q_scale, kv_scale)
```

Inputs must be contiguous. Scales may be device FP32 scalars or `[1]` vectors.
The plan preallocates O/LSE and 31.75 MiB split scratch, supports caller-provided
output/stream/workspace, and reuses them across calls. Calls sharing a plan or
scratch must be ordered. Outputs are overwritten by the next call.
DCP8 merging requires LSE; no external max-score output is produced.
`return_lse=False` omits LSE for standalone O-only callers.
This bundle provides the callable kernel; it does not install a vLLM runtime
adapter or fuse MLA into K2.

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
Random token-level physical permutations were also measured. The benchmark
prints all samples and can save them with `--output`.

Each graph captures 40 complete calls; each timing replays it 20 times. Nine
rounds rotate and reverse implementation order, with medians reported. Outputs
are poisoned after capture and checked after replay to reject empty graphs.
Both paths use the same data, scales, causal mask and required outputs. AITER
head padding and metadata construction are excluded from timing. These are
warm GPU graph timings, excluding compilation, allocation and host overhead.

## Run the benchmark

Requirements: MI355X gfx950, ROCm PyTorch, FlyDSL and AITER with CPRR MLA.
The measured image supplies these dependencies:

```text
vllm/vllm-openai-rocm@sha256:520e098f54ec96837228aaddea0a833cbf6374dbdffd480c1d65f26ffab7701b
```

Versions: Torch `2.12.0+rocm10.0.0`, HIP `7.15.26333`, FlyDSL `0.3.4.1`, AITER
from the image; MI355X has 256 CUs. A vLLM installation is not needed to run
these two Python files. Keep them in the same directory.

From this checkout's root in that environment:

```bash
uv venv --python 3.12 --system-site-packages .venv
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py
```

Defaults cover 8192/8200/9216/65536/65544/262144 global tokens, both paged and
random layouts, rank 0, seed 1234, nine timing rounds and ASM split caps
64/128/256/512. No input configuration file is needed. For selected lengths:

```bash
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py \
  --lengths 8192 65536 262144 --layouts paged \
  --asm-splits 64 128 256 512 --output /tmp/kimi-mla.json
```

To reproduce the independent rank/seed confirmation:

```bash
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py \
  --lengths 8200 65544 262152 --rank 7 --seed 2026 \
  --asm-splits 64 128 256 --output /tmp/kimi-mla-confirm.json
```

Each selected shape is checked against FP32 O/LSE before timing. The command
prints environment, kernel hash, errors and per-round timings; `--output`
optionally saves JSON. This performance harness accepts lengths >=8192, where
it applies `atol=0.003, rtol=0.03` to O and `atol=2e-4, rtol=2e-5` to LSE.
The kernel itself also supports shorter valid lengths.

## Validation and limits

The measured kernel passed 111 validation records covering all eight ranks,
dynamic lengths from 8 through 262144, split thresholds, page mappings, NaN
padding, nonunit scales, multiple seeds and simulated eight-rank DCP merging.
O-only output was also checked. The compact benchmark includes the ordinary
input O/LSE reference checks; it does not rerun that entire validation suite.

At Q/K standard deviation 0.5, error is close to ASM. At standard deviation 1.5,
maximum absolute error matched ASM on three tested seeds, but RMS error was
about 10-13% higher. Typical LSE maximum error is approximately `2e-6`.
Full K3 engine TP8 TPOT and model quality evaluation have **not** been run;
the reported speedups are for the single-rank MLA O+LSE operator.

The GPU measurements precede this file consolidation. The GPU builder and
input/reference/baseline functions retain the measured ASTs. The standalone
benchmark has been compiled and linted, but has not been rerun on GPU since
consolidation. Default device selection uses
`torch.accelerator.current_device_index()`.

Sources: [PyTorch PR #194309](https://github.com/pytorch/pytorch/pull/194309)
head `5bc1e0fd3e4e89f762a5b6cc805814ff8b64f156`, and
[vLLM PR #60683](https://github.com/vllm-project/vllm/pull/60683) for K3 shapes
and CPRR semantics. The branch is based on `989829c0beb9d5270fbf4c560565c0212b7d0d88`.
The originally measured kernel SHA256 is
`864f7488249f8df6778b3f8e786c9ce1a6f14705a039b2d3e6f0c257070bae22`;
formatting and packaging change the file hash, not the GPU builder AST.
AI assistance was used for implementation and benchmark preparation.
