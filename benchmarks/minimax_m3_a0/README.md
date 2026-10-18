[English](README.md) | [中文](README_zh.md)

# MiniMax-M3: eight independent A0 patches

This branch is a **portable ablation patch package**, not an all-optimizations-on
engine. The vLLM source remains at
`2a02f6efe319c885e3ccbcecde402e0028f9ec1e`. Select exactly one patch per clean
A0 source tree or image. Do not apply the eight overlays sequentially: several
intentionally modify the same files with different independent experiments.

Start with [the handover prompt](HANDOVER.md).

## Contents

| Arm | Standalone patch | Scope and dependency boundary |
|---|---|---|
| R1 | [Compile/preparation](arms/r01_compile_prepare/vllm.patch) | A1 compile plus attention preparation; a combined topic |
| R2 | [Router](arms/r02_router/vllm.patch) | A4 BF16-input/FP32 router GEMM, no later router variants |
| R3 | [Draft metadata](arms/r03_draft_metadata/vllm.patch) | A5 metadata reuse/multistep capability, no sharded sampling |
| R4 | [Sampling](arms/r04_sampling/vllm.patch) | A33 target and draft sharded greedy sampling, no R3 |
| R5 | [Sparse PA](arms/r05_sparse_pa/vllm.patch) | A38 partition alignment, native A0 merge |
| R6 | [Merge/score](arms/r06_score_merge/vllm.patch) | A39 grouped merge v2 plus A41 cap432; **native A0 partitions**, no R5 |
| R7 | [GEMM1](arms/r07_gemm1/vllm.patch) | A48 TN64 G1 with A0 token/slot-layout adapter; **A0 CK-Tile G2 retained** |
| R8 | [GEMM2/TP](arms/r08_gemm2_tp/vllm.patch) | A37 G2/TP and sort-clear reuse, A0 slot-input adapter; **A0 G1 retained** |

Each directory includes an exact `payload/`, a SHA256 manifest and the original
plan/audit as provenance. `archived_plan.json` may mention the old host and
historical warmed A0; it is **not** the current testing protocol.
`protocol.json` and the handover take precedence.

The patches include non-Python runtime assets: R6's two `.cpp.jinja` templates,
and R8's two `.so` libraries. Do not transfer only `*.py`.
The R8 libraries require the pinned CPython/PyTorch/ROCm ABI; native source and
portability notes are under `probes/`. Rebuilding them changes the tested
artifact and requires a new manifest and numerical validation.

## Verification status

- All 381 exported A0 vLLM files match the base Git commit; no source bootstrap
  patch is needed within that audited scope.
- The image audit covers392 exported vLLM/AITER files; this is a scoped audit,
  not a claim that every byte in the full image has been independently audited.
- All eight standalone binary patches were CPU-tested against clean A0 source
  files and matched their payload hashes. Overlay application rejects a dirty
  or already-patched A0.
- Previous R1/R2 runs completed240/240 requests and real C24/C4 smoke, but their
  historical-A0 performance comparison is invalid because of warmup differences.
  Earlier R1/R2/R3 results also had a capture-list mismatch.
- **No valid independent eight-arm performance result is claimed.**
  R7/R8's new slot-layout adapters still require GPU numerical validation.
  The packaged harness/probe adaptations have not run a GPU benchmark here.

`environment.json` records the observed source environment.
`VALIDATION.json` records packaging checks; it is not a performance result.

## Use a source checkout

Create a clean worktree at the base commit. Set `PACKAGE` to this directory
and `PY` to a Python3.12 virtual-environment interpreter:

```bash
git worktree add --detach /path/to/clean-r01 2a02f6efe319c885e3ccbcecde402e0028f9ec1e
"$PY" "$PACKAGE/tools/apply_arm.py" --target /path/to/clean-r01 \
  --arm r01_compile_prepare --source-only --check-only
git -C /path/to/clean-r01 apply "$PACKAGE/arms/r01_compile_prepare/vllm.patch"
```

`--source-only` validates vLLM files, **not the external AITER/runtime
dependencies**. The supported reproducible runtime is the exact A0 image
described below, with an individual overlay applied.

## Use the exact runtime

The base image is local, not a public registry tag:

```text
local/mm3-vllm-2a02f6e:latest
sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7
```

Transfer it with `docker image save`/`docker image load`; see the handover.
The Git bundle contains neither this11.46 GB uncompressed image nor model
weights. Do not silently substitute a recent vLLM/AITER image.

Build a common A0 wrapper and a selected arm:

```bash
"$PY" tools/build_image.py --arm a0 --tag local/mm3-handoff-a0:zero
"$PY" tools/build_image.py --arm r01_compile_prepare \
  --tag local/mm3-handoff-r01:zero
```

The Dockerfile creates a `uv` virtual environment using the base image's
system packages; it installs or upgrades no package. Runtime Python is
`/opt/mm3/.venv/bin/python`. Image builds do not access GPU devices.

## New measurement protocol

**Every 8k/1k E2E, including the destination's one A0 baseline, explicitly uses
`--num-warmups 0`.** This is the user's latest instruction. Do not send smoke,
microbenchmark or other completion requests to that formal server beforehand.
Normal engine initialization and graph capture remain enabled.

Use one exclusively allocated MI355X TP4 group. Measure A0 once, then each
correctness-valid independent arm once and reuse that A0. Use fresh processes
and separate fresh writable cache directories for every formal E2E.
Keep negative results. No0.15 ms gate, no reruns to select the fastest result,
no cumulative-baseline promotion, no AgenticX performance and no C4 performance.

Use `tools/run_arm.py` with the destination model/cache paths, GPU indices,
device-group GID and a shared allocation lock. It validates effective capture
sizes and key runtime settings before formal requests. `--phase smoke` uses
a **different** real-Eagle3 server and collects C24/C4 traces.
Inspect `client.log` and `server.log`; do not dump detailed result JSON.

After checking correctness and optimized dispatch, compare:

```bash
"$PY" tools/compare_results.py --baseline /results/a0 \
  --candidate /results/r01 --output /results/r01/comparison.json
```

The comparison requires a matching same-host/GPU, zero-warmup A0 contract.
A matching workload alone does not prove correctness or actual kernel dispatch.
Independent percentages are not additive.

## CPU verification

```bash
cd tools
"$PY" -B -m unittest -v test_handoff
```

Do not run the old `prepare.py` scripts: their donor-directory prerequisites
are not the portable runtime. The checked-in patches/payloads are the
immutable products of that preparation.
