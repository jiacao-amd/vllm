# Accepted N18: portable MiniMax-M3 verification

**English** | [中文](README_zh.md)

This bundle transfers the accepted **28-file vLLM overlay**, not the historical
eight independent A0 ablations. The cumulative stack is
**N02 + N05 + N06 + N07 (including N03 sampling) + N12 + N16 + N18**.
N16 replaces N14's G1 configuration. **AITER is unchanged.** The optimization
goal remains paused; this handoff does not claim a 9% gain.

## Handover prompt

> Verify this exact accepted N18 branch on one other compatible machine.
> Read both manifests and the frozen protocol before running anything.
> Build the source overlay on the pinned official image; do not rebuild vLLM
> or replace AITER, kernels, model weights, or benchmark dependencies.
> Use the same four gfx950 GPUs for both arms, with exclusive ownership and
> identical engine/client settings. Prepare each arm's compiler cache using
> only the frozen native dummy workload. Run the standard Eagle3 C24/C4 smoke
> in separate processes and empty private caches; inspect all four ranks'
> traces for the retained implementation, not just successful text output.
> Qualify each arm before its formal E2E; run exactly one N0 and one N18 E2E:
> 240 requests, C24, 8192/1024, and **`--num-warmups 0`**.
> Reuse this machine's N0 denominator. Preserve every log, failure, source/image
> hash, GPU UUID, cache audit and raw result. Do not seed caches from smoke or
> E2E, subtract JIT/stalls, select favorable requests, or rerun a negative result.
> Report throughput and all saved latency percentiles; stop on identity,
> correctness or dispatch failures. Do not resume optimization or change code.

## Pinned environment and prior evidence

| Item | Pin / contract |
| --- | --- |
| Official base | `vllm/vllm-openai-rocm@sha256:c6cb4ab021e31691f63a5757278153fbe728a71c722651a4549ddbb5ac3067ab` |
| vLLM base commit | `1b2a666391f2a24228a1436b64d547c34cc6b6ce` |
| Runtime | Python `3.12.15`; vLLM `0.32.0rc1.dev369+g1b2a66639.rocm72`; Torch `2.13.0+git733fca1`; HIP `7.2.53211` |
| Kernel packages | Triton `3.8.0+git669b31ac`; AITER `0.1.25`; FlyDSL `0.3.4.1` |
| Hardware | Four MI355X / `gfx950` GPUs, TP4; record physical UUIDs and keep selection identical across arms |
| Source machine (historical) | `smci355-ccs-aus-m15-17`, GPUs4–7, UID:GID13807:13807, supplemental device group110; not a required remote username/group |
| Models | Main snapshot `229447a2d64149ffb45e454ebeb74b7a9a45ce7d`; draft snapshot `96692486b5fd38ebf8fd2a5f6bb53427d30819a8` |
| Precision | Original MXFP4 model, BF16 activations, original FP32 router weights, FP8 main/index KV; no precision conversion |
| Engine | Block128, memory utilization0.9, chunked prefill/prefix cache enabled, token budget32768, max sequences48; HTTP concurrency is24 |
| Capture | `FULL_DECODE_ONLY`, exact **37 sizes** in [protocol.py](protocol.py); max128 alone is not equivalent |
| Formal workload | 240 requests, fixed8192 input/1024 output tokens, C24, seed0, chat template, ignoreEOS, stream interval20; Eagle3 K3, synthetic acceptance2.78, thinking enabled |
| Separate smoke | Original standard (non-synthetic) Eagle3 C24/C4 checks, thinking disabled; not timing or a cache-preparation workload |
| Protocol ID | `official-nightly-tp4-c24-8k1k-http0-dummy-compile-v1` |

Source-machine results below are **historical single runs**, not remote baselines:

| Arm | Output tok/s | Change |
| --- | ---: | --- |
| Original official N0 | 2414.2174046171085 | Denominator on the source machine only |
| Accepted N18 | 2546.872092705771 | +5.49472834696% vs N0; +0.6675924329% vs N16 |

N18 versus N16 had mixed latency: mean TTFT **+3.5898%**, p99 ITL **+2.3961%**;
mean TPOT **−1.1724%**, p99 E2EL **−1.2353%**, p99.9 ITL **−3.0054%**.
P99 ITL was **+0.7218% versus N0**. Keep the two observed runtime-JIT warning
families and their time; no subtraction or statistical-significance claim.
N06 retains an additional588MiB/GPU packed head plus fallback weights.
N18 exactly reconstructs the checked router weights from BF16 terms; this
does **not** claim bitwise equivalence of BF16x3 and native FP32 dot products.

## What the old R topics became

This is a topic map, not byte-identical ports or additive performance credit.

| Old topic | Accepted nightly disposition |
| --- | --- |
| R1 / A1 attention preparation | Native N0 already has guarded fused norm/RoPE/cache preparation. Old compile wiring is not fully reproduced or counted again. |
| R2 / A4 router | N01's standalone E2E stayed rejected; its range extension is retained through accepted N12. N18 adds a separate exact-M96 BF16x3 router; original weights and other-M fallbacks remain. |
| R3 / A5 draft metadata | N07 implements the current metadata-update ABI and fused draft loop, not an old-source overwrite. |
| R4 / A33 target/draft sampling | Standalone N03 stayed rejected; its sampling implementation is retained inside accepted N07. |
| R5 / A38 sparse PA | N05 partitions the actual single-KV-head path and retains native merge/fallbacks. |
| R6 / A39+A41 merge/score | N02 retains the score-cap residual. The old grouped-merge patch is not separately installed or credited. |
| R7 / A48 G1 | Current-layout integration came through N14; N16 selects exact-M96 TN64/TK128/w2 instead of N14's TK256. This is not the old A48 payload. |
| R8 / A37 G2+TP | **Not retained.** N13 fusion/transport/publication experiments did not qualify; native G2/TP remains. |

N06's packed draft head is an additional accepted nightly topic, not one of
those eight old arms. Never apply the old R patches on top of this bundle.

## Bundle and build

[accepted_manifest.json](accepted_manifest.json) pins the28 final source hashes
and required N18 flags. [native_manifest.json](native_manifest.json) pins the
official source/package inventory. `install.py` verifies the native preimage and
accepted overlay; the Dockerfile uses the **repository root** as build context.
The frozen `protocol.py`, `prepare_cache.py`, `compile_extension.py`,
`model_check.py`, `reference_run_e2e.py` and `client/*.py` preserve the original
workload. Use `run_inside.py` as the portable entry, **not**
`reference_run_e2e.py`'s original host launcher.
The accepted source has historical GPU evidence; the new packaging/wrapper has
CPU/build checks only, not another GPU measurement or remote validation.

**This is a reproduction branch, not a merge-ready upstream PR.** Read-only
typo, root-import, forbidden-boolean, config and path checks passed, but upstream
mypy reported **46 errors in6 files** (including dynamic FlyDSL classes/attributes
and Optional narrowing). The forbidden-import check also flags the direct Triton
import in the exact frozen `gpu_pack.py`. Full upstream static checks are **not
clean**; no type, import or formatting cleanup was made during handoff.

The new installer/wrapper passed **7 CPU tests, Ruff and the full584-source image
audit**. These checks do not imply that all pre-commit hooks passed. The planned
handoff commit will record specific known hook skips to preserve the measured
payload bytes; upstream cleanup remains separate. No GPU retesting was performed
for this handoff.

Commands below are for the receiving operator; no GPU/Docker execution is
performed by writing these instructions. From the checked-out repository root:

```bash
set -euo pipefail
set -o noclobber
REPO="$PWD"
BUNDLE="$REPO/benchmarks/minimax_m3_nightly"
BASE='vllm/vllm-openai-rocm@sha256:c6cb4ab021e31691f63a5757278153fbe728a71c722651a4549ddbb5ac3067ab'
CANDIDATE='mm3-n18-handoff:20261019'
docker pull "$BASE"
docker build -f "$BUNDLE/Dockerfile" -t "$CANDIDATE" "$REPO"
docker image inspect "$BASE" "$CANDIDATE"
```

Keep the build log and resolved image IDs. The rebuilt N18 image can have a new
image ID; it must match this bundle's source hashes, not the source machine's
historical image ID.

## Model mounts, ownership and execution

Set absolute host paths below. `MAIN_CACHE` and `DRAFT_CACHE` are model-specific
Hugging Face cache roots containing `snapshots/` **and all referenced blobs**.
Snapshot files may be symlinks: copying only `snapshots/<hash>` is insufficient.
If links reach a shared parent `blobs/`, mount that tree at the corresponding
container path as well. Alternatively, provide fully materialized snapshot
directories at the exact paths below; do not silently choose another revision:

```text
/models/main/snapshots/229447a2d64149ffb45e454ebeb74b7a9a45ce7d
/models/draft/snapshots/96692486b5fd38ebf8fd2a5f6bb53427d30819a8
```

The example assumes all links resolve within each mounted model cache root.
Resolve this before loading either model. Ensure model/bundle files are readable
by the chosen non-root UID/GID, and all result/cache directories are writable.
The original run used UID:GID13807:13807 and device group110; use the receiving
machine's actual render-device group instead of assuming110.

```bash
MAIN_CACHE=/absolute/path/models--amd--MiniMax-M3-MXFP4
DRAFT_CACHE=/absolute/path/models--Inferact--MiniMax-M3-EAGLE3-GQA
WORK=/absolute/path/fresh-mm3-n18-verification
HIP_GPUS=4,5,6,7                 # Or four available local IDs; same for both arms.
RUN_UID=$(id -u)
RUN_GID=$(id -g)
RENDER_GID=$(getent group render | cut -d: -f3)
test "$RUN_UID" -ne 0
test -n "$RENDER_GID"
mkdir "$WORK"                  # Its parent must exist; refuse an old run root.
```

Hold the machine's agreed GPU lock across the sequence and ensure no competing
job uses these GPUs. Execute serially. The helper below starts a fresh non-root
container and temporary system-site-packages venv for each stage; it installs
no packages. `/suite` and `/client` are read-only, `/results` must be fresh,
and only the stage's private `/cache` is writable.

```bash
run_stage() {
  local arm=$1 stage=$2 cache=$3 out=$4
  local image=$BASE
  local image_id
  local -a prepared=()
  if [ "$arm" = n18 ]; then image=$CANDIDATE; fi
  image_id=$(docker image inspect --format '{{.Id}}' "$image") || return
  if [ "$stage" = e2e ]; then
    prepared=(--mount "type=bind,src=$WORK/$arm/prepare,dst=/prepared,readonly")
  fi
  mkdir "$out" || return        # Refuse an existing result directory.
  docker run --rm --user "$RUN_UID:$RUN_GID" --group-add "$RENDER_GID" \
    --device /dev/kfd --device /dev/dri --ipc host --network host --workdir /tmp \
    -e MM3_HANDOFF_IMAGE_ID="$image_id" \
    -e HIP_VISIBLE_DEVICES="$HIP_GPUS" -e OMP_NUM_THREADS=1 \
    -e HOME=/cache/home -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e UV_CACHE_DIR=/tmp/mm3-uv \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONNOUSERSITE=1 -e PYTHONPATH=/suite \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --mount "type=bind,src=$BUNDLE,dst=/suite,readonly" \
    --mount "type=bind,src=$BUNDLE/client,dst=/client,readonly" \
    --mount "type=bind,src=$MAIN_CACHE,dst=/models/main,readonly" \
    --mount "type=bind,src=$DRAFT_CACHE,dst=/models/draft,readonly" \
    --mount "type=bind,src=$cache,dst=/cache" \
    --mount "type=bind,src=$out,dst=/results" "${prepared[@]}" \
    --entrypoint sh "$image_id" -c \
    'uv venv --offline --python /usr/bin/python3 --system-site-packages /tmp/mm3-venv &&
     exec /tmp/mm3-venv/bin/python -B /suite/run_inside.py "$@"' \
    mm3-handoff --stage "$stage" --arm "$arm" --port 6101 \
    > "$WORK/$arm/$stage.console.log" 2>&1
}
```

`run_inside.py` applies/checks the frozen environment and arm-specific flags;
do not enable N18 flags on N0 or change the accepted configuration.
`MM3_HANDOFF_IMAGE_ID` is mandatory in **every stage** and comes from Docker
inspection; run the same resolved ID. Keep port6101 free. Console logs are outside
the mounted result directories; `status.json` records completion.
The temporary uv cache is outside `/cache`, so bootstrap cannot seed the
compiler template. Do not pre-create `/cache/home` before the empty-cache check.

### Prepare → smoke → one E2E per arm

Run **N0 prepare → N0 smoke → N0 E2E**, then the same sequence for N18.
**Stop if any stage fails.** Preparation and smoke start with separate empty
caches. Only formal E2E starts from a clone of its arm's prepared template.

```bash
for arm in n0 n18; do
  mkdir "$WORK/$arm"
  mkdir "$WORK/$arm/cache_template"
  run_stage "$arm" prepare "$WORK/$arm/cache_template" "$WORK/$arm/prepare" || exit 1
  mkdir "$WORK/$arm/cache_smoke"
  run_stage "$arm" smoke "$WORK/$arm/cache_smoke" "$WORK/$arm/smoke" || exit 1
  test ! -e "$WORK/$arm/cache_e2e"
  cp -a "$WORK/$arm/cache_template" "$WORK/$arm/cache_e2e"
  run_stage "$arm" e2e "$WORK/$arm/cache_e2e" "$WORK/$arm/e2e" || exit 1
done
```

Preparation must pass `compile_ready.json`; smoke must pass C24/C4
`model_check.json` and all-rank dispatch checks. `cache_fingerprint.json` in the
preparation output binds the cache bytes checked before E2E. The template must
remain exactly as preparation left it; never copy smoke-generated files back.
Normal compile-time/native graph warmups are separate from HTTP warmups.

Each formal E2E must complete240/240 requests with1966080 input and245760 output
tokens, the exact37 capture sizes, and **zero HTTP warmups**. `/prepared` is the
matching arm's read-only preparation output. Do not share caches between arms
or use the smoke/E2E cache as a new template. Preserve failures; investigate
before authorizing any new attempt.
If N0 has already completed under this exact package/protocol on the receiving
machine, retain its result and run only the outstanding N18 stages; do not rerun
N0 merely because candidate setup failed.

Report `100 * (remote_N18_output_tok_s / remote_N0_output_tok_s - 1)` using
only these same-machine runs. Include mean/p90/p99/p99.9 TTFT, TPOT, ITL and E2EL,
completion/token totals, image/source pins, GPU UUIDs, cache audits and raw logs.
With stream interval20, client ITL is a streamed-chunk interval, not per-token
kernel latency. A successful smoke or faster component is not E2E acceptance;
one remote pair is verification evidence, not statistical significance.
