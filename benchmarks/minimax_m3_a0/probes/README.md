[English](README.md) | [中文](README_zh.md)

# Portable R7/R8 numerical probe

This is an **untimed, synthetic TP4 correctness check**, not a model evaluation
or an E2E admission by itself. Run each arm in its own freshly started process
with only that arm installed over the pinned A0 runtime. No GPU test or native
build was performed while packaging this driver.

## Run on the destination

Set `PACKAGE` to `benchmarks/minimax_m3_a0` (or its copied package directory),
`PY` to the matching Python 3.12 virtual-environment interpreter, `MM3_IMAGE_ID`
to the actual selected-arm image ID, and `EXPECTED_UUIDS` to a JSON list of the
four destination GPUs' `str(torch.cuda.get_device_properties(i).uuid)` values,
in local-rank order. Obtain those identities independently; do not copy the
source machine's UUIDs. Reserve that GPU group exclusively before running.

```bash
export PYTHONDONTWRITEBYTECODE=1
export HIP_VISIBLE_DEVICES=0,1,2,3  # Replace with four destination GPU indices.
unset ROCR_VISIBLE_DEVICES
"$PY" -B -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$PACKAGE/probes/correctness_probe.py" \
  --arm R7 --image-id "$MM3_IMAGE_ID" \
  --expected-gpu-uuids "$EXPECTED_UUIDS" \
  --output-dir /separate/results/r7-numerical
```

Use `--arm R8` in a separate R8 runtime and a fresh output directory for R8.
The package root defaults to the driver's parent directory's parent; use
`--package-root` if mounting the script separately. Outputs must be outside
the package. Existing `correctness_probe*.json` evidence is never overwritten.
The probe requires four unique declared HIP indices, actual world size 4,
four visible GPUs per rank, gfx950, and exact rank-ordered UUID equality.
Reports retain expected/observed UUIDs, image ID, Torch/HIP versions, installed
package roots, and source/manifest hashes. The image ID is runner-declared;
the outer runner remains responsible for verifying it.

## Preserved numerical contract

The original shared driver is pinned at SHA256
`299fe7b479c9fbaa826823d4d7c3603c5ffb7694c6e9f8440077a8b6fa08f8b0`.
Its complete numerical body is AST-identical after packaging: the same
synthetic MXFP4 weights, 32 changing-route replays, zero-input case, bounded
rank skew, actual BM16/token-slot check, 0.005 L2 gates, exact owner-cyclic TP
reference, R8 bitwise G1 check, counters/epochs, and dispatch gates.

G1/local/metadata clones occur **inside each check graph and before native TP**;
global output is cloned after TP. Live references are retained separately for
poisoning, and every saved buffer must be disjoint from all saved/live buffers
across both arms. Reference replay finishes before candidate live poisoning.
These validation copies are deliberately inside correctness graphs; no timing
result is produced. The native communicator pool remains 128 MiB.

The driver checks the selected package/installed payload against its
`arms/.../patch_manifest.json`, and unchanged reference primitives against
`baseline_manifest.json`. It does not replace the package's full overlay audit.
An interrupted run can still require process-level cleanup; the vendored
helper attempts every registered cleanup action without replacing a primary
error, but cannot recover from a GPU hang or partially constructed native owner.

CPU-only validation:

```bash
"$PY" -B -m unittest discover -s "$PACKAGE/probes" -p test_probe_snapshots.py -v
ruff check --no-cache "$PACKAGE/probes" \
  --per-file-ignores 'cleanup_protocol.py:I001,cleanup_protocol.py:E501'
```

The style exceptions preserve the byte-exact cleanup helper. No Torch/device
module is imported by these tests. The tests also pin the numerical-body AST
and the three vendored files.

## Runtime dependencies and minimum transferable set

Transfer the package (manifests, selected arm's **entire** payload, and these
probes) plus the exact A0 runtime image or equivalently pinned native runtime.
The vLLM branch alone is insufficient. There are no imports from the original
experiment directories or source-host absolute paths.

- AITER: `fused_moe`, `ops.flydsl.moe_kernels`,
  `kernels.moe_2stage_a16wmix` G1/G2/utils, `mxfp4_gemm_common`, `buffer_ops`,
  `layout_utils`, `tensor_shim`, and R8 `communication_ops_utils`; native
  CK-Tile/Opus sorting and stages, quantization/shuffle, activation, and
  `CustomAllreduce` must remain available.
- FlyDSL: compiler, expression/typing/ROCDL/runtime modules and compiled MLIR
  libraries; matching ROCm Torch, Gloo, Triton, and AITER native modules.
- R7 includes the private exact A48 activation helper. Global A0 `act.py`
  remains unchanged. R7 does not import A37 G2/TP; R8 does not import R7 G1.
- R8 includes both pinned IPC/view `.so` files; copying only Python files is
  insufficient. The source-only files below are not compiled by the probe.

Recorded runtime pins are in `../environment.json`: vLLM
`0.28.1rc1.dev628+g2a02f6efe.rocm723`, Python `3.12.13`, Torch
`2.12.0+git6bbd260` (C++11 ABI), HIP `7.2.53211`, ROCm `7.2.3`,
AITER `0.1.21.post2`, FlyDSL `0.3.2`, and Triton `3.7.1+gitf0b55c07`.
Use its wheel SHA256 records too; version labels are not sufficient.

## R8 native-source provenance and rebuild recipe

Both sources are byte-identical to A37 `build_v2/package`. Workspace ownership
originates in A36; the non-owning tensor view originates in A37. They expose
IPC ownership/metadata and a BF16 view at byte offset 16384, shape `[96,6144]`;
they contain no numerical kernels. Keep all ownership and cleanup unchanged.

| File | SHA256 |
|---|---|
| `native/workspace.cu` | `3e3e68d91f75a1e086049d838814af2b694cf2cf33902043c2beed8d5faaa1aa` |
| `native/tensor_view.cu` | `2289907d7c4219fbe414f481c3e48286e5bd477ebb302883702ecbc7440e0311` |
| frozen `mm3_a36_workspace.so` | `df4c721282a849db04852c036bcd017146feea2d02dc065aa2fd52b71ba5d5bf` |
| frozen `mm3_a37_tensor_view.so` | `6b4d5dd8aa04bee3568c4117b3ebcae58c75c1954f4bb7b489cf7a9fb556f603` |
| `cleanup_protocol.py` | `8c0afe6a8847c184efa2e9852ca245486cf0fdde304c72c1da92b6534720b53a` |

For an explicitly authorized reconstruction, the original extension recipe
is below. `BUILD_ROOT` must be a separate writable directory. This compiles and
loads host extensions, but calls none of their allocation/device APIs. It is
**not** part of the probe and has not been executed here.

```bash
export PACKAGE PYTORCH_ROCM_ARCH=gfx950
export BUILD_ROOT=/separate/native-rebuild
"$PY" -B - <<'PY'
import os
from pathlib import Path
from torch.utils.cpp_extension import load

sources = Path(os.environ["PACKAGE"]) / "probes" / "native"
root = Path(os.environ["BUILD_ROOT"])
for name, source in (
    ("mm3_a36_workspace", "workspace.cu"),
    ("mm3_a37_tensor_view", "tensor_view.cu"),
):
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    module = load(
        name=name,
        sources=[str(sources / source)],
        build_directory=str(destination),
        extra_cuda_cflags=["-O2", "-std=c++17", "--offload-arch=gfx950"],
        extra_cflags=["-O2"],
        verbose=False,
    )
    print(name, module.__file__)
PY
```

The original recipe uses `/opt/rocm/bin/hipcc`, Torch/Python headers and
Torch/HIP libraries. Frozen ELF dependencies include `libamdhip64.so.7`,
`libc10`, `libtorch_cpu`, `libtorch_python`, and the C++ runtime; no RPATH is
embedded. A rebuild may differ bytewise even with identical sources.
**Do not replace frozen payload binaries or relax the probe's SHA checks.**
A changed binary needs a separately approved build/installation manifest and
fresh numerical validation; this package's default path is the exact binaries
in the exact A0 ABI.

All destination 8k/1k E2Es, including the single reusable remote A0 baseline,
must use `--num-warmups 0`. This probe runs separately and is not HTTP warmup.
The historical warmed A0 result is not the denominator for that protocol.
