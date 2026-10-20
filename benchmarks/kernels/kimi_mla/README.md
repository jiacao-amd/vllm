# Kimi-K3 conc1 MLA on MI355X

A standalone three-file bundle for conc1, seven speculative tokens, TP8/DCP8
and FP8 KV cache:

- [mla.py](mla.py): FlyDSL `KimiMLADecodePlan` and
  `KimiMLAPrefillPlan` APIs for decode and prefill.
- [benchmark.py](benchmark.py): input generation, FP32 reference, AITER ASM
  baselines, original PR194309 FlexAttention forward, and alternating O+LSE
  graph timings. Select prefill with `--mode prefill`.
- [README.md](README.md): shapes, optimizations, results and usage.

## Decode API

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
from mla import KimiMLADecodePlan

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

## Decode optimizations

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

## Decode recorded performance

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

## Decode benchmark

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

## Decode validation and limits

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

The decode performance table predates file consolidation. The GPU builder and
input/reference/baseline functions retain the measured ASTs. After adding
prefill, the standalone decode benchmark passed an 8192-token, paged-cache,
rank-0 GPU smoke test, including FP32 O/LSE checks and graph replay. That smoke
test used ASM split cap 256; it does not replace the best-cap measurements
above. Default device selection uses `torch.accelerator.current_device_index()`.

Sources: [PyTorch PR #194309](https://github.com/pytorch/pytorch/pull/194309)
head `5bc1e0fd3e4e89f762a5b6cc805814ff8b64f156`, and
[vLLM PR #60683](https://github.com/vllm-project/vllm/pull/60683) for K3 shapes
and CPRR semantics. The branch is based on `989829c0beb9d5270fbf4c560565c0212b7d0d88`.
The originally measured kernel SHA256 is
`864f7488249f8df6778b3f8e786c9ce1a6f14705a039b2d3e6f0c257070bae22`;
formatting and packaging change the file hash, not the GPU builder AST.
AI assistance was used for implementation and benchmark preparation.

## Prefill API

K3 TP8 prefill uses one request with **12 heads per rank**, decompressed
Q/K `[S,12,192]` and V `[S,12,128]`, after `kv_b_proj`. This differs from the
compressed decode layout above. `KimiMLAPrefillPlan` supports dense causal
attention with Q length equal to KV length, for multiples of 128 from 128 to
262144. It returns BF16 O `[S,12,128]` and natural-log FP32 LSE `[S,12]`.
The softmax scale is `192**-0.5`; there is no max-score output.

```python
from mla import KimiMLAPrefillPlan

# Q/K/V are contiguous BF16 tensors on the same gfx950 device.
plan = KimiMLAPrefillPlan(seq_len=q.shape[0], device=q.device)
out, lse = plan(q, k, v)
# Optional: plan(q, k, v, output=caller_output, lse=caller_lse, stream=stream)
```

The precision path matches K3's existing FP8 ASM prefill: Q/K/V are converted
with scale 1 to E4M3FN, P uses scaled FP8, accumulation is FP32, and O is BF16.
The original FlexAttention baseline uses BF16 arithmetic. The numerical error
metrics below distinguish these precision choices.

For inputs already in FP8, construct the plan with
`input_dtype=torch.float8_e4m3fn`. BF16 calls with S <= 512 convert inside the
attention kernel; longer calls use one fused Q/K/V conversion kernel followed
by attention. The latter preallocates 6 KiB of FP8 scratch per token, or 48 MiB
at 8192 tokens. Reuse a plan across ordered layers/calls, or pass a shared
`workspace=(q_fp8, k_fp8, v_fp8)` to plans with the same shape. Calls do not
allocate tensors or read device scalars on the host. Warm up before capture.

This API covers the PR's equal-length prefill fast path. The caller performs
KV decompression and provides contiguous tensors. Chunked-context merging and
paged-cache lookup belong to the surrounding backend. A full 64k/256k causal
benchmark below is a single synthetic prefill call; an engine with chunked
prefill can instead process the prompt in smaller calls. This three-file
bundle provides the callable kernel without a vLLM runtime adapter.

## Prefill optimizations

- Native gfx950 FP8 MFMA for QK192 and PV128, operating on 12 heads directly.
  AITER's prefill wrapper pads them to 16.
- Q stays in registers; K/V use separate LDS regions with double buffering.
  The next tile is written to LDS after QK to shorten register lifetimes.
- The K layout uses 16-byte groups, with transposed LDS reads for V.
  Fully visible KV tiles skip per-element causal comparisons.
- A 128-token KV tile amortizes softmax and barriers for longer requests.
  Four-wave query tiles are used below 32768 tokens and eight-wave tiles at or
  above that threshold. The short BF16 path uses a 64-token KV tile.
- The factor 256 for FP8 P is incorporated into the exponent offset and
  normalization, removing per-element probability scaling multiplies.
- At S >= 32768, a wave ballot skips accumulator rescaling when none of its
  rows increases the running softmax maximum. This preserves the exact running
  maximum and the existing FP8 precision path.
- Long BF16 calls fuse all three conversions into one launch, with no head
  padding or output-cropping copy. Query blocks are dispatched in reverse
  order to run the longer causal blocks first.

These choices were selected with GPU measurements. They do not make every
short shape faster than BF16 FlexAttention: the 128/512/2048-token cases below
remain slower when input conversion is included.

## Prefill benchmark

The same fixed image and library versions listed above were used on
`crsuse2-m2m-200`, MI355X GPU 4. Inputs are BF16, seed 1234, Q/K standard
deviation 0.5. All times include O and LSE. Seven rounds rotate/reverse
implementation order; the table reports medians in **microseconds**.

| Tokens | BF16 Flex | FlyDSL, BF16 inputs | ASM full | ASM with cat padding |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 7.226 | 8.281 | 48.907 | 35.848 |
| 512 | 13.987 | 16.710 | 73.484 | 57.482 |
| 2048 | 43.448 | 51.714 | 142.048 | 110.890 |
| 8192 | 260.703 | 209.758 | 455.618 | 372.516 |
| 65536 | 13917.940 | 9891.673 | 12378.914 | 11736.698 |
| 262144 | 1289402.246 | 153429.272 | 193447.900 | 191025.720 |

`flydsl` includes any input conversion. `asm_full` reproduces PR60683's
`repeat(...).contiguous()` padding, FP8 casts, scale creation, ASM, reduction,
and output cropping. `asm_full_cat` is an improved wrapper using concatenation
padding and a preallocated scale. Both preparation-inclusive ASM results are
reported so padding overhead can be assessed separately.

With inputs already converted and padded as required, the kernel comparison is:

| Tokens | FlyDSL, FP8 inputs | ASM, FP8 inputs | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | 7.172 | 14.056 | 1.960x |
| 512 | 13.389 | 23.691 | 1.770x |
| 2048 | 43.462 | 48.441 | 1.115x |
| 8192 | 187.447 | 200.759 | 1.071x |
| 65536 | 9696.067 | 10338.338 | 1.066x |
| 262144 | 153080.017 | 185611.267 | 1.213x |

An independent run with seed 2026, Q/K standard deviation 1.5, and five
alternating rounds confirmed the long-shape gains (microseconds):

| Tokens | FlyDSL, BF16 inputs | ASM full | FP8 kernel speedup vs ASM |
| ---: | ---: | ---: | ---: |
| 8192 | 213.235 | 453.716 | 1.045x |
| 65536 | 10127.126 | 12349.377 | 1.044x |
| 262144 | 158606.909 | 193426.978 | 1.171x |

The 8k/64k kernel gains are modest: about 4-7% in these two runs. These are
single-GPU operator measurements, not full-model TTFT, TPOT, or quality results.
At 256k, the original Flex forward's generic paired sparse-mask bookkeeping
has substantial runtime and compilation cost; the large speedup includes
eliminating that work for dense causal input.

Run the full comparison using the two Python files in this directory:

```bash
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py --mode prefill \
  --lengths 128 512 2048 8192 65536 262144 --samples 7 \
  --output /tmp/kimi-prefill.json
```

The benchmark downloads two unchanged source files from PR194309 commit
`5bc1e0fd3e4e89f762a5b6cc805814ff8b64f156` into `~/.cache/kimi_mla/` and
verifies both SHA256 hashes. Pass `--flex-source /path/to/source_directory`
for an existing offline copy. Initial compilation of the original 256k
baseline can take several minutes; compilation and metadata construction are
outside GPU timing. Defaults stop at 64k. Every graph is checked for complete
O/LSE writes after poisoning both outputs.

For exhaustive short-shape checks and higher score magnitudes:

```bash
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py --mode prefill \
  --lengths 128 256 512 1024 2048 --impls flydsl asm --full-reference \
  --seed 2026 --magnitude 2.0 --samples 1
```

## Prefill validation and precision

GPU validation passed for:

- Fifteen exhaustive short cases: S = 128/256/512/1024/2048, Q/K standard
  deviations 0.5/1.5/2.0, seed 2026, all query rows and heads.
- The final long path at S = 32768, standard deviation 2.0, seed 3033, checking
  **every query row** against FP32 reference for both FlyDSL and ASM.
- Both timing runs above, using sampled causal/tile-boundary rows (14 rows for
  the long shapes), all heads and all visible KV tokens. Full output/LSE
  finiteness is checked after graph capture and replay.
- Fourteen API cases including storage offsets, output/LSE guard canaries,
  nondefault streams, two plans sharing conversion scratch, BF16/FP8 input,
  uniform attention, and captured replay after input changes. S = 8320/32896
  cover a final 128-row query tile on the corresponding launch paths.
- Invalid shape/dtype/device/contiguity rejection, original-source hash
  verification, and a fresh pinned-source download plus GPU smoke test.

To repeat the full 32k reference check:

```bash
.venv/bin/python benchmarks/kernels/kimi_mla/benchmark.py --mode prefill \
  --lengths 32768 --impls flydsl asm --full-reference \
  --seed 3033 --magnitude 2.0 --samples 1
```

The FP32 reference uses FP8-rounded Q/K/V for ASM and the new kernel. The
original Flex implementation first rounds `Q * scale * log2(e)` to BF16;
its implementation check reproduces that step. All paths additionally report
error against the original BF16-input FP32 formula. LSE is checked with
`atol=0.001, rtol=0.0001`. For FP8 P, the output checks use normalized L2 < 4%
and maximum per-row relative L2 < 8%, avoiding misleading elementwise relative
errors near zero. Maximum absolute, RMS, LSE, and BF16-reference errors are
included in the JSON output.

At 8k/64k/256k with standard deviation 0.5, the new kernel and ASM have
approximately 2.9-3.0% normalized L2 error against the original BF16-input
reference, versus 0.12-0.13% for BF16 Flex. In the full 32k stress check at
standard deviation 2.0, that error is 10.00% for the new kernel and 10.01% for
ASM; against FP8-rounded inputs it is 0.86% and 1.03%, respectively. The new
kernel's maximum LSE error against the FP8 reference is 0.000542 in that check.

The new kernel is not uniformly more accurate than ASM. In the short
1024-token, standard-deviation-2.0 case, its RMS error against the FP8
reference is 14.5% higher than ASM; including input quantization, RMS error
against the BF16 reference is within 0.1% of ASM. The FP8 path follows the
existing ASM precision tradeoff and has larger error than BF16 FlexAttention.
Operator checks do not establish model quality.

The final prefill timings and checks used kernel SHA256
`1acc883e54821595e9f55343b2c27d2734a1218ec53df6c4ae8d8adab1a950cd`.
No K3 runtime integration or full-model evaluation is included.
