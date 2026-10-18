[English](README.md) | [中文](README_zh.md)

# 可移植的 R7/R8 数值检查

这是**不计时、使用合成数据的 TP4 正确性检查**，不是模型评估，也不能单独作为
E2E 准入依据。每个实验臂都必须在独立的新进程中运行，运行环境只在固定 A0 上
安装对应的一项补丁。本次打包未执行 GPU 检查或原生扩展构建。

## 在目标机器运行

将 `PACKAGE` 指向 `benchmarks/minimax_m3_a0` 或复制后的包目录，`PY` 指向匹配的
Python 3.12 虚拟环境解释器，`MM3_IMAGE_ID` 设为所选实验臂的实际镜像 ID。
`EXPECTED_UUIDS` 为目标机器四张 GPU 的
`str(torch.cuda.get_device_properties(i).uuid)` 值组成的 JSON 列表，顺序必须与
本地 rank 一致。请独立确认设备身份，不要沿用源机器 UUID；运行前需独占该 GPU 组。

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

R8 使用独立的 R8 运行环境、`--arm R8` 和新的输出目录。包根目录默认为脚本目录的
上级；单独挂载脚本时可用 `--package-root` 指定。输出必须位于包目录之外，已有的
`correctness_probe*.json` 证据不会被覆盖。检查要求显式声明四个不同的 HIP 设备
编号，实际 world size 为 4，每个 rank 可见四张 GPU，架构为 gfx950，且观测到的
UUID 与声明列表逐项一致。报告记录声明及实际 UUID、镜像 ID、Torch/HIP 版本、
实际包路径及源码/清单哈希。镜像 ID 由启动器声明，外层启动器仍需验证其真实性。

## 保持不变的数值约束

原共享驱动的固定 SHA256 为
`299fe7b479c9fbaa826823d4d7c3603c5ffb7694c6e9f8440077a8b6fa08f8b0`。
打包后的完整数值主体与其 AST 一致：保留相同的合成 MXFP4 权重、32 次变化路由
回放、零输入、有限 rank 到达偏移、实际 BM16/token-slot 检查、0.005 L2 阈值、
按 owner 循环顺序执行的精确 TP 参考、R8 G1 逐位检查、计数器/epoch 和实际调度检查。

G1/local/元数据的克隆在**每个检查图内部、原生 TP 之前**完成，全局输出在 TP 后
克隆。保留独立的实时输出引用用于填充毒值，所有保存缓冲区必须与双方全部保存及
实时缓冲区不重叠。候选实时缓冲区仅在参考回放完成后填毒。这些验证复制有意放在
正确性图内，不产生性能结果；原生通信缓冲池仍为 128 MiB。

驱动使用 `arms/.../patch_manifest.json` 检查所选包内及已安装的 payload，并用
`baseline_manifest.json` 检查未改变的 A0 参考原语。它不能替代包的完整覆盖层审计。
中断后仍可能需要进程级清理；复制的清理 helper 会尝试全部已注册动作且保留主异常，
但无法恢复 GPU 挂起或只完成部分构造的原生资源所有者。

仅 CPU 验证：

```bash
"$PY" -B -m unittest discover -s "$PACKAGE/probes" -p test_probe_snapshots.py -v
ruff check --no-cache "$PACKAGE/probes" \
  --per-file-ignores 'cleanup_protocol.py:I001,cleanup_protocol.py:E501'
```

上述格式例外用于保留清理 helper 的原始字节。测试不导入 Torch 或设备模块，
同时固定数值主体 AST 和三个复制文件的哈希。

## 运行依赖与最小迁移集合

需要迁移整个包（清单、所选实验臂的**完整** payload 和本目录），以及原始 A0
运行镜像或等价且完整固定的原生运行环境。仅迁移 vLLM 分支不够；本检查已移除
对原实验目录和源机器绝对路径的导入依赖。

- AITER：`fused_moe`、`ops.flydsl.moe_kernels`、
  `kernels.moe_2stage_a16wmix` 的 G1/G2/utils、`mxfp4_gemm_common`、
  `buffer_ops`、`layout_utils`、`tensor_shim`，以及 R8 的
  `communication_ops_utils`；还需保留原生 CK-Tile/Opus 排序和阶段内核、
  量化/重排、激活及 `CustomAllreduce`。
- FlyDSL：编译器、表达式/类型/ROCDL/运行时模块和编译后的 MLIR 库，以及匹配的
  ROCm Torch、Gloo、Triton、AITER 原生模块。
- R7 自带逐字节一致的 A48 私有激活 helper，全局 A0 `act.py` 不变。
  R7 不导入 A37 G2/TP，R8 不导入 R7 G1。
- R8 包含固定的 IPC/view 两个 `.so`，只复制 Python 文件不够。下面的原生源码
  仅供重建依据，检查驱动不会编译它们。

`../environment.json` 记录了环境版本：vLLM
`0.28.1rc1.dev628+g2a02f6efe.rocm723`、Python `3.12.13`、
Torch `2.12.0+git6bbd260`（C++11 ABI）、HIP `7.2.53211`、ROCm `7.2.3`、
AITER `0.1.21.post2`、FlyDSL `0.3.2`、Triton `3.7.1+gitf0b55c07`。
还应核对其中的 wheel SHA256；仅版本号相同不足以保证一致。

## R8 原生源码来源与重建命令

两份源码与 A37 `build_v2/package` 逐字节一致。Workspace 所有权实现来自 A36，
非拥有型 tensor view 来自 A37。它们只提供 IPC 所有权/元数据接口，以及偏移
16384 字节、形状 `[96,6144]` 的 BF16 视图，不含数值内核。必须保持所有权和清理
语义不变。

| 文件 | SHA256 |
|---|---|
| `native/workspace.cu` | `3e3e68d91f75a1e086049d838814af2b694cf2cf33902043c2beed8d5faaa1aa` |
| `native/tensor_view.cu` | `2289907d7c4219fbe414f481c3e48286e5bd477ebb302883702ecbc7440e0311` |
| 固定 `mm3_a36_workspace.so` | `df4c721282a849db04852c036bcd017146feea2d02dc065aa2fd52b71ba5d5bf` |
| 固定 `mm3_a37_tensor_view.so` | `6b4d5dd8aa04bee3568c4117b3ebcae58c75c1954f4bb7b489cf7a9fb556f603` |
| `cleanup_protocol.py` | `8c0afe6a8847c184efa2e9852ca245486cf0fdde304c72c1da92b6534720b53a` |

只有在明确授权重建后，才使用以下原始扩展构建方式。`BUILD_ROOT` 必须是独立的可写
目录。该命令编译并加载主机扩展，但不调用其分配/设备 API；它**不属于**检查流程，
本次未执行。

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

原构建使用 `/opt/rocm/bin/hipcc`、Torch/Python 头文件以及 Torch/HIP 库。
固定 ELF 依赖包括 `libamdhip64.so.7`、`libc10`、`libtorch_cpu`、
`libtorch_python` 和 C++ 运行库，未嵌入 RPATH。即使源码相同，重建后二进制字节
也可能不同。**不要覆盖冻结的 payload 二进制，也不要放宽检查中的 SHA 约束。**
不同二进制需要单独批准的构建/安装清单和新的数值验证；本包默认使用原始 A0 ABI
下的原始二进制。

目标机器上所有 8k/1k E2E，包括只测一次并复用的 A0 基线，都必须使用
`--num-warmups 0`。本检查独立运行，不构成 HTTP 预热；历史有预热的 A0 结果不能
作为新协议的对照基线。
