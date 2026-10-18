[English](HANDOVER.md) | [中文](HANDOVER_zh.md)

# Handover prompt for the destination agent

You are taking over **testing**, not further optimization. Test the eight
independent MiniMax-M3 patches in this branch on one destination machine.
Read `README.md`, `protocol.json`, `environment.json`, the selected arm's
manifest, and the relevant kernel implementation before starting GPU work.

## Objective and current authority

For each arm R1–R8, compare **A0 + that one patch** against one freshly measured
unoptimized A0 on the destination. Do not compare against the previous arm,
apply all patches together, resume A68, create a PR, or promote accepted A48.

Latest user instruction: **no additional request warmup for any 8k/1k E2E**.
Use explicit `--num-warmups 0` for A0 and every arm. Do not send a health-check
completion, smoke request, microbenchmark request, or other hidden warmup
to the formal server. Ordinary `/health` checks, engine initialization and
normal CUDA/HIP graph capture are allowed.

Use the same TP4 GPU group, software, model revisions, protocol and client.
Measure A0 once and reuse it. Each correctness-valid arm gets one E2E,
including flat or negative micro results. No0.15 ms acceptance gate and no
repeated runs to select a better number. Preserve failures and all artifacts.
Run microbenchmarks and correctness separately from the formal E2E process.
No AgenticX performance. C4 is correctness only.

## Transfer and base identity

The source vLLM branch is:

```text
codex/minimax-m3-a0-eight-patches-20261018
```

The bundle contains that branch's commits after this prerequisite:

```text
2a02f6efe319c885e3ccbcecde402e0028f9ec1e
```

Use an existing vLLM clone containing that commit, then:

```bash
git bundle verify /transfer/minimax-m3-a0-eight-patches.bundle
git fetch /transfer/minimax-m3-a0-eight-patches.bundle \
  codex/minimax-m3-a0-eight-patches-20261018:codex/minimax-m3-a0-eight-patches-20261018
git worktree add /path/to/mm3-handoff codex/minimax-m3-a0-eight-patches-20261018
cd /path/to/mm3-handoff/benchmarks/minimax_m3_a0
```

Do not switch or overwrite a dirty destination worktree.
This is a patch-package branch; its engine source is intentionally still A0.

The exact runtime must be transferred separately. **The local image tag is
not downloadable from Docker Hub.** On the source machine, outside Git:

```bash
docker image inspect local/mm3-vllm-2a02f6e:latest --format '{{.Id}}'
docker image save -o /transfer/mm3-a0-runtime.tar local/mm3-vllm-2a02f6e:latest
sha256sum /transfer/mm3-a0-runtime.tar
```

Transfer the archive and checksum, then on the destination:

```bash
sha256sum /transfer/mm3-a0-runtime.tar
docker image load -i /transfer/mm3-a0-runtime.tar
docker image inspect local/mm3-vllm-2a02f6e:latest --format '{{.Id}}'
```

Required image ID:
`sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7`.
Stop if it differs. Do not substitute `latest`, rebuild a different dependency
stack silently, or assume version labels guarantee binary compatibility.
The image is approximately11.46 GB uncompressed; weights are separate.
The archive was **not** generated as part of patch packaging.

## Environment and model data

The source machine has8 MI355X GPUs (`gfx950`), about287.98 GiB visible memory
per GPU, two EPYC9575F64-core CPUs/256 logical CPUs, kernel6.8.0-90-generic,
and amdgpu driver6.16.13. Recent runs selected HIP GPUs4–7. Use four suitable
**exclusively allocated** destination GPUs; do not copy source GPU UUIDs.

Pinned container:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| vLLM | 0.28.1rc1.dev628+g2a02f6efe.rocm723 |
| ROCm / Torch HIP | 7.2.3 / 7.2.53211 |
| PyTorch | 2.12.0+git6bbd260, C++11 ABI enabled |
| Triton | 3.7.1+gitf0b55c07 |
| AITER | amd-aiter0.1.21.post2 |
| FlyDSL | 0.3.2 |
| Transformers | 5.20.0 |

Target: `amd/MiniMax-M3-MXFP4`, revision
`229447a2d64149ffb45e454ebeb74b7a9a45ce7d`.
Draft: `Inferact/MiniMax-M3-EAGLE3-GQA`, revision
`96692486b5fd38ebf8fd2a5f6bb53427d30819a8`.

`--model-root` and `--draft-root` mean HF repository cache directories containing
`snapshots/<revision>/` and the corresponding blobs. Preserve symlink targets.
The source target root was `/data/models/models--amd--MiniMax-M3-MXFP4`;
the draft root was
`/home/jiacao/.cache/mm3-eagle3/models--Inferact--MiniMax-M3-EAGLE3-GQA`,
with separate blobs in `/home/jiacao/.cache/mm3-eagle3/blobs`.
If retaining that layout, pass `--draft-blobs` on the destination.
Container paths stay `/models/main`, `/models/draft`, `/models/blobs`.

Use the destination non-root UID/GID and render/video group.
Source values13807:13807 and110 are **not** portable defaults.
Never leave root-owned result files. Use one shared allocation lock for all
jobs on the same GPU group; this helper's lock cannot reserve GPUs against
uncooperative jobs. Do not stop any unrelated container.
The runner sets `HOME`, `USER` and `LOGNAME` explicitly; keep them when running
manual commands under a numeric UID absent from the image's passwd database.
If teardown cannot be confirmed, the lock's `.owner.json` marker remains and
blocks the next launch. Inspect that exact owned container before clearing
the marker; a log-write failure is not permission to leave a server running.

## Fixed serving and benchmark settings

Keep TP4, BF16 activations/MXFP4 weights, FP8 KV, block128, C24,
8192 input/1024 output tokens,240 formal requests, seed0, unlimited request
rate, `--ignore-eos`, `--use-chat-template`, Eagle3 K3 and synthetic acceptance2.78.
No precision, acceptance or model-computation shortcuts.

Keep `ROCM_AITER_UNIFIED_ATTN`, `--moe-backend aiter`,
`indexer_kv_dtype=fp8`, prefix caching, chunked prefill,
max batched tokens32768, max sequences48, stream interval20,
GPU-memory-utilization0.9, and `FULL_DECODE_ONLY`.
Preserve the complete37 capture-size list in `a0_runtime_config.json`,
not merely `max_cudagraph_capture_size=128`.

`tools/run_arm.py` supplies the original AITER/shared-expert/quick-reduce
environment and the arm's switches from its manifest. It uses fresh per-run
writable caches. Do not omit the R4/R5/R6/R8 experiment switches.
R6 specifically requires `MM3_A39_MERGE_VARIANT=v2`, not its v1 default.
Audit-only R6 launch logging is enabled in smoke, not timed E2E.

The four client files are pinned copies of InferenceX, not `vllm bench serve`.
Keep them together, including the unconditionally imported `encoding_dsv4.py`.
`--save-detailed` retains per-request evidence after the formal interval.
Before formal requests, the runner hashes the client and scoped engine files
inside the actual container. It also hashes the target's `tokenizer.json`;
identical revision names/configurations alone are insufficient.

## Runbook

Create/use a Python3.12 venv with `uv` on the destination for the host helpers.
They use the standard library. Set `PY` to that venv interpreter.
Container Python is `/opt/mm3/.venv/bin/python`; do not assume the original
A0 image has `/opt/vllm-test/.venv`.

First run the CPU patch checks:

```bash
cd tools
"$PY" -B -m unittest -v test_handoff
cd ..
```

Build one unoptimized wrapper and a selected arm:

```bash
"$PY" -B tools/build_image.py --arm a0 --tag local/mm3-handoff-a0:zero
"$PY" -B tools/build_image.py --arm r01_compile_prepare \
  --tag local/mm3-handoff-r01:zero
```

Use destination-specific variables, for example:

```bash
COMMON=(--model-root "$MODEL_ROOT" --draft-root "$DRAFT_ROOT" \
  --gpus "$ALLOCATED_GPUS" --device-group "$RENDER_GID" \
  --lock "$SHARED_GPU_LOCK" --port 6101)
# Add --draft-blobs "$DRAFT_BLOBS" when the snapshot layout requires it.
```

In a separate process/container, validate kernels and real-Eagle3 C24/C4:

```bash
"$PY" -B tools/run_arm.py --arm r01_compile_prepare \
  --image local/mm3-handoff-r01:zero --output "$RESULTS/r01-smoke" \
  --phase smoke "${COMMON[@]}"
"$PY" -B tools/audit_dispatch.py --arm r01_compile_prepare \
  --run-dir "$RESULTS/r01-smoke"
```

Before R7/R8 smoke, run the portable numerical probes described in
`probes/README.md`. Before the other E2Es, review relevant correctness/micro
coverage below. CPU patch checks do not substitute for GPU correctness.

Run one fresh A0, then each independent arm once:

```bash
"$PY" -B tools/run_arm.py --arm a0 --image local/mm3-handoff-a0:zero \
  --output "$RESULTS/a0" "${COMMON[@]}"
"$PY" -B tools/run_arm.py --arm r01_compile_prepare \
  --image local/mm3-handoff-r01:zero --output "$RESULTS/r01" "${COMMON[@]}"
"$PY" -B tools/compare_results.py --baseline "$RESULTS/a0" \
  --candidate "$RESULTS/r01" --output "$RESULTS/r01/comparison.json"
```

Repeat the candidate build/check/E2E for R2–R8, always reusing `$RESULTS/a0`.
Do not reuse a smoke server for E2E. The helper refuses existing result
directories; do not delete them to rerun. Preserve JIT and cold-first-wave
effects as part of the requested zero-warmup result; do not discard the first
wave or secretly warm until the number improves.

## Required correctness and attribution review

| Arm | Check before attributing performance |
|---|---|
| R1 | Eager/compiled target outputs, hidden/aux states and cache writes; fused QK-preparation dispatch on all ranks |
| R2 | BF16-input/FP32-output router at actual M96 and draft shapes; routing/top-k unchanged; `_rocm_fp32_router_gemm_kernel` |
| R3 | Three draft steps, positions/lengths/slots/block boundaries/finished requests; fused-step log present, no metadata-rebuild fallback |
| R4 | Target and draft greedy equivalence, ties/vocab padding/fallbacks; all four sharded sampling kernel families appear |
| R5 | A0 versus R5 PA with threshold/empty-row/page-ID/scale cases and unchanged KV; `mm3_a38_paged_attention_head1_hybrid` |
| R6 | Native A0 partitions/partials retained, merge outputs match; both Jinja templates installed; cap432 verified by `[1,24,18]` launch record on four ranks |
| R7 | A0's actual BM16 and BF16 `[96,5,768]` slot intermediate; only G1 changes; unchanged CK-Tile G2; private TN64 kernel dispatch |
| R8 | A0 G1 bitwise unchanged; slot-input G2, poison/epoch/counter/IPC checks; one TP reduction, no extra sort/layout kernel, correct fallback |

R5/R6's prior donor preparation tests are not standalone remote tests.
Their old GPU fixtures need adapting to installed R5/R6, and the R6 reference
must be native A0, **not** A38. Do not claim the package already completed
that GPU validation. Use the current implementations to build an isolated
kernel microbenchmark if archived fixtures are unavailable. Include full
component cost, not just a shortened kernel fragment.

R7/R8 include necessary **new** A0 layout adapters, not literal unmodified
A48/A37 kernels. Their GPU numerical validation is mandatory and unfinished.
The private R7 activation helper intentionally differs from global A0 AITER;
do not replace the global helper. R8 includes two exact native libraries;
do not rebuild them with a different PyTorch ABI and retain the old hash claim.

## Existing evidence and final deliverable

Historical A0 was2310.4520822142695 tok/s with48 HTTP warmups and GPU selection0–3.
Early standalone R1/R2/R3 runs had a19-versus37 capture-list mismatch.
Corrected-list R1/R2 later ran with zero HTTP warmups, while historical A0
was warmed: their apparent−6.53%/+0.91% deltas are **not independent gains**.
Do not use these historical numbers as the destination baseline or acceptance target.

This handover performed packaging/CPU checks, not a new E2E or kernel speed test.
The original cumulative A48 was not changed.

Deliver one table with destination A0 and each arm's image/patch hashes,
correctness and actual-dispatch status, microbenchmark result, output
throughput, mean TPOT, P99 TTFT/ITL/E2EL and delta versus the shared new A0.
Keep failed/flat/negative rows. Link raw artifacts; distinguish numerical
differences from causal or statistically significant claims.
Do not add the eight percentages or automatically promote a cumulative branch.
