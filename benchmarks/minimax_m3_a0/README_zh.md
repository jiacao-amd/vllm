[English](README.md) | [中文](README_zh.md)

# MiniMax-M3：八项独立 A0 补丁

本分支是**可移植的独立消融补丁包**，不是默认同时启用八项优化的引擎。
vLLM 源码保持在 `2a02f6efe319c885e3ccbcecde402e0028f9ec1e`。
每次在干净的 A0 源码或镜像上只选择一项。不要顺序叠加八份 overlay：
其中多项有意修改同一个文件，用于不同的独立实验。

请先阅读[交接提示词](HANDOVER_zh.md)。

## 内容

| 项目 | 独立补丁 | 范围与依赖边界 |
|---|---|---|
| R1 | [Compile／前处理](arms/r01_compile_prepare/vllm.patch) | A1 compile 与 attention 前处理，作为一个组合主题 |
| R2 | [Router](arms/r02_router/vllm.patch) | A4 BF16 输入／FP32 router GEMM，不含后续 router 方案 |
| R3 | [Draft metadata](arms/r03_draft_metadata/vllm.patch) | A5 metadata 复用／多步能力，不含分片采样 |
| R4 | [采样](arms/r04_sampling/vllm.patch) | A33 target 与 draft 分片贪心采样，不含 R3 |
| R5 | [Sparse PA](arms/r05_sparse_pa/vllm.patch) | A38 分区对齐，保留原生 A0 merge |
| R6 | [Merge／score](arms/r06_score_merge/vllm.patch) | A39 grouped merge v2 与 A41 cap432；**保留 A0 分区**，不含 R5 |
| R7 | [GEMM1](arms/r07_gemm1/vllm.patch) | A48 TN64 G1，适配 A0 token/slot 布局；**保留 A0 CK-Tile G2** |
| R8 | [GEMM2／TP](arms/r08_gemm2_tp/vllm.patch) | A37 G2／TP 和 sort-clear 复用，适配 A0 slot 输入；**保留 A0 G1** |

每个目录都包含原样保存的 `payload/`、SHA256 清单，以及用于溯源的旧计划和审计。
`archived_plan.json` 可能仍提到旧机器或历史带预热的 A0，**不代表当前测试协议**。
以 `protocol.json` 和交接提示词为准。

补丁包含非 Python 运行时资产：R6 的两个 `.cpp.jinja` 模板，以及 R8 的两个
`.so` 库。不要只传输 `*.py`。R8 库依赖固定的 CPython／PyTorch／ROCm ABI；
原生源码及可移植性说明放在 `probes/`。重编译会改变被测产物，必须更新清单
并重新做数值验证。

## 验证状态

- 导出的 381 个 A0 vLLM 文件均与基础 Git 提交一致；在该审计范围内无需额外
  源码引导补丁。
- 镜像审计覆盖 392 个导出的 vLLM／AITER 文件；这是限定范围的审计，
  不代表已经逐字节独立审计整个镜像。
- 八份独立二进制补丁均已在 CPU 上验证：可应用到干净 A0，结果与 payload
  哈希一致。应用工具拒绝脏的或已经打过补丁的 A0。
- 之前 R1/R2 完成过 240/240 请求和真实 C24/C4 smoke，但与历史 A0 的性能
  比较因预热不一致而无效；更早的 R1/R2/R3 还存在 capture 列表不一致。
- **目前不宣称已获得八项独立优化的有效性能结论。**
  R7/R8 新增 slot 布局适配仍需 GPU 数值验证。
  本次打包后的启动器／probe 适配没有在这里运行 GPU 基准测试。

`environment.json` 记录源机器的实测环境信息。
`VALIDATION.json` 记录打包检查，不是性能结果。

## 在源码工作树中使用

在基础提交上建立干净 worktree。将 `PACKAGE` 设置为本目录，
将 `PY` 设置为 Python 3.12 虚拟环境解释器：

```bash
git worktree add --detach /path/to/clean-r01 2a02f6efe319c885e3ccbcecde402e0028f9ec1e
"$PY" "$PACKAGE/tools/apply_arm.py" --target /path/to/clean-r01 \
  --arm r01_compile_prepare --source-only --check-only
git -C /path/to/clean-r01 apply "$PACKAGE/arms/r01_compile_prepare/vllm.patch"
```

`--source-only` 只验证 vLLM 文件，**不验证外部 AITER／运行时依赖**。
可复现的受支持运行方式是使用下述原始 A0 镜像，再独立应用一项 overlay。

## 使用固定运行时

基础镜像是本地镜像，不是公开 registry tag：

```text
local/mm3-vllm-2a02f6e:latest
sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7
```

通过 `docker image save`／`docker image load` 传输，命令见交接提示词。
Git bundle 不包含这个未压缩约 11.46 GB 的镜像，也不包含模型权重。
不要擅自换成近期的 vLLM／AITER 镜像。

构建公共 A0 包装镜像和一项候选：

```bash
"$PY" tools/build_image.py --arm a0 --tag local/mm3-handoff-a0:zero
"$PY" tools/build_image.py --arm r01_compile_prepare \
  --tag local/mm3-handoff-r01:zero
```

Dockerfile 使用基础镜像的 system packages 创建 `uv` 虚拟环境，不安装或
升级任何包。运行时解释器为 `/opt/mm3/.venv/bin/python`。
镜像构建不访问 GPU 设备。

## 新的测量协议

**每次 8k/1k E2E（包括目标机器上唯一的一次 A0 baseline）都显式使用
`--num-warmups 0`。** 这是用户最新要求。正式服务实例在计时前不得接收
smoke、微基准或其他 completion 请求；服务正常初始化和 graph capture 保持开启。

使用同一组独占的 MI355X TP4 GPU。A0 只测一轮；随后每项正确性通过的独立
候选各测一轮，共用这个 A0。每轮正式 E2E 都用全新进程和独立、空的可写缓存目录。
保留负向结果，不设 0.15 ms 门槛，不反复测试挑最快结果，不自动更新累计 baseline，
不测 AgenticX 性能，也不测 C4 性能。

使用 `tools/run_arm.py`，传入目标机器的模型／缓存路径、GPU 编号、设备组 GID
以及所有占用该 GPU 组的任务共用的锁。正式请求前会验证有效 capture 列表
和关键运行参数。`--phase smoke` 使用**另一个**真实 Eagle3 服务实例，
采集 C24/C4 trace。查看 `client.log`、`server.log`，不要直接输出完整详细 JSON。

检查正确性及实际优化调度后，再做比较：

```bash
"$PY" tools/compare_results.py --baseline /results/a0 \
  --candidate /results/r01 --output /results/r01/comparison.json
```

比较工具要求同机器、同 GPU 组、零预热的 A0 contract 匹配。
工作负载匹配本身不能证明正确性或实际走到优化 kernel。
各项独立百分比不能相加。

## CPU 检查

```bash
cd tools
"$PY" -B -m unittest -v test_handoff
```

不要直接运行旧 `prepare.py`：这些脚本依赖 donor 目录，不是可移植运行时的一部分。
已提交的 patch／payload 才是该准备过程的固定产物。
