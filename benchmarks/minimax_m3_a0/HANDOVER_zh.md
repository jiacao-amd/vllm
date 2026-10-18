[English](HANDOVER.md) | [中文](HANDOVER_zh.md)

# 给目标机器 Agent 的交接提示词

你接手的是**测试工作**，不是继续搜索优化。在一台目标机器上测试本分支的
八份独立 MiniMax-M3 补丁。启动 GPU 工作前，阅读 `README_zh.md`、
`protocol.json`、`environment.json`、所选项的 manifest 及相关 kernel 实现。

## 目标与当前授权

R1–R8 每项都比较 **A0＋该项唯一补丁** 与目标机器上新测的未优化 A0。
不要与上一项比较，不要叠加八项，不恢复 A68、不创建 PR、不替换已接受的 A48。

用户最新要求：**所有 8k/1k E2E 都不做额外请求预热**。
A0 和所有候选均显式设置 `--num-warmups 0`。不要在正式服务实例上提前发送
健康检查 completion、smoke、微基准请求或其他隐式预热。
普通 `/health` 检查、引擎初始化和正常 CUDA/HIP graph capture 可以保留。

所有测试使用同一 TP4 GPU 组、软件、模型 revision、协议和客户端。
A0 只测一轮，之后复用。每项正确性通过的候选各做一轮 E2E，包括微基准持平
或负向的项。不设 0.15 ms 门槛，不重复测量挑更好数字。保留失败及全部产物。
微基准和正确性测试必须与正式 E2E 分进程／容器执行。
不测 AgenticX 性能；C4 只做正确性。

## 传输与基础版本

源 vLLM 分支为：

```text
codex/minimax-m3-a0-eight-patches-20261018
```

Bundle 包含该分支在以下前置提交之后的提交：

```text
2a02f6efe319c885e3ccbcecde402e0028f9ec1e
```

在已包含该提交的 vLLM clone 中执行：

```bash
git bundle verify /transfer/minimax-m3-a0-eight-patches.bundle
git fetch /transfer/minimax-m3-a0-eight-patches.bundle \
  codex/minimax-m3-a0-eight-patches-20261018:codex/minimax-m3-a0-eight-patches-20261018
git worktree add /path/to/mm3-handoff codex/minimax-m3-a0-eight-patches-20261018
cd /path/to/mm3-handoff/benchmarks/minimax_m3_a0
```

不要切换或覆盖目标机器上有未提交改动的工作树。
这是补丁包分支，其引擎源码有意保持为 A0。

精确运行时需要单独传输。**本地镜像 tag 不能从 Docker Hub 下载。**
在源机器上、Git 目录以外执行：

```bash
docker image inspect local/mm3-vllm-2a02f6e:latest --format '{{.Id}}'
docker image save -o /transfer/mm3-a0-runtime.tar local/mm3-vllm-2a02f6e:latest
sha256sum /transfer/mm3-a0-runtime.tar
```

传输归档和校验值后，在目标机器执行：

```bash
sha256sum /transfer/mm3-a0-runtime.tar
docker image load -i /transfer/mm3-a0-runtime.tar
docker image inspect local/mm3-vllm-2a02f6e:latest --format '{{.Id}}'
```

镜像 ID 必须为：
`sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7`。
不一致则停止；不要用 `latest` 替代、擅自重建另一套依赖，
也不要认为版本号一致就保证二进制兼容。
该镜像未压缩约 11.46 GB，模型权重另行传输。
本次补丁打包**没有生成**运行时镜像归档。

## 环境与模型数据

源机器为 8 张 MI355X（`gfx950`），每张可见显存约 287.98 GiB，
双路 EPYC 9575F 64 核 CPU／256 逻辑 CPU，内核 `6.8.0-90-generic`，
amdgpu 驱动 `6.16.13`。近期测试选择 HIP GPU 4–7。
目标机器使用四张合适且**独占分配**的 GPU，不要复制源机器 GPU UUID。

固定容器环境：

| 组件 | 版本 |
|---|---|
| Python | 3.12.13 |
| vLLM | 0.28.1rc1.dev628+g2a02f6efe.rocm723 |
| ROCm／Torch HIP | 7.2.3／7.2.53211 |
| PyTorch | 2.12.0+git6bbd260，开启 C++11 ABI |
| Triton | 3.7.1+gitf0b55c07 |
| AITER | amd-aiter 0.1.21.post2 |
| FlyDSL | 0.3.2 |
| Transformers | 5.20.0 |

Target：`amd/MiniMax-M3-MXFP4`，revision：
`229447a2d64149ffb45e454ebeb74b7a9a45ce7d`。
Draft：`Inferact/MiniMax-M3-EAGLE3-GQA`，revision：
`96692486b5fd38ebf8fd2a5f6bb53427d30819a8`。

`--model-root` 和 `--draft-root` 指包含 `snapshots/<revision>/` 和对应
blobs 的 HF 仓库缓存目录，必须保留软链接目标。
源 target 根目录为 `/data/models/models--amd--MiniMax-M3-MXFP4`；
源 draft 根目录为
`/home/jiacao/.cache/mm3-eagle3/models--Inferact--MiniMax-M3-EAGLE3-GQA`，
另有 `/home/jiacao/.cache/mm3-eagle3/blobs`。
若保留该布局，目标机器需传入 `--draft-blobs`。
容器内路径保持 `/models/main`、`/models/draft`、`/models/blobs`。

使用目标机器的非 root UID/GID 和 render/video 设备组。
源值 `13807:13807` 和 `110` **不能直接作为可移植默认值**。
不要留下 root 所有的结果文件。所有占用同一 GPU 组的任务必须使用同一分配锁；
辅助脚本的锁无法阻止不配合的其他任务使用 GPU。不要停止无关容器。
启动器显式设置 `HOME`、`USER` 和 `LOGNAME`；使用镜像 passwd 中不存在的数字
UID 手动运行命令时，也必须保留这些变量。若无法确认容器已退出，锁旁的
`.owner.json` 会保留并阻止下一次启动。清除此标记前，先检查其中记录的那个
自有容器；不能因为写日志失败就留下服务继续运行。

## 固定服务与基准测试配置

保持 TP4、BF16 activation／MXFP4 weight、FP8 KV、block128、C24、
8192 输入／1024 输出 token、240 个正式请求、seed0、无限请求速率、
`--ignore-eos`、`--use-chat-template`、Eagle3 K3 和 synthetic acceptance2.78。
不得改变精度、acceptance 或减少模型计算。

保持 `ROCM_AITER_UNIFIED_ATTN`、`--moe-backend aiter`、
`indexer_kv_dtype=fp8`、prefix caching、chunked prefill、
max batched tokens32768、max sequences48、stream interval20、
GPU-memory-utilization0.9 和 `FULL_DECODE_ONLY`。
必须保留 `a0_runtime_config.json` 中完整的 37 项 capture-size 列表，
不能只设置 `max_cudagraph_capture_size=128`。

`tools/run_arm.py` 提供原来的 AITER／shared-expert／quick-reduce 环境变量，
以及各项 manifest 中的开关；每轮使用新的独立可写缓存。
不要漏掉 R4/R5/R6/R8 实验开关。
R6 必须设置 `MM3_A39_MERGE_VARIANT=v2`，不能使用默认 v1。
R6 launch 审计日志只在 smoke 中开启，不在计时 E2E 中开启。

四个客户端文件是固定的 InferenceX 副本，不是 `vllm bench serve`。
必须一起使用，包括无条件导入的 `encoding_dsv4.py`。
`--save-detailed` 在正式计时结束后保留逐请求证据。
正式请求之前，启动器会校验实际容器中的客户端和限定范围内引擎文件的哈希，
并记录 target 的 `tokenizer.json` 哈希；仅 revision 名和配置相同还不够。

## 执行步骤

目标机器使用 `uv` 创建或复用 Python 3.12 venv 来运行 host 辅助工具，
这些工具只用标准库。将 `PY` 设置为该 venv 的解释器。
容器解释器为 `/opt/mm3/.venv/bin/python`；不要假定原始 A0 镜像有
`/opt/vllm-test/.venv`。

先运行 CPU 补丁检查：

```bash
cd tools
"$PY" -B -m unittest -v test_handoff
cd ..
```

构建未优化的包装镜像和一项候选：

```bash
"$PY" -B tools/build_image.py --arm a0 --tag local/mm3-handoff-a0:zero
"$PY" -B tools/build_image.py --arm r01_compile_prepare \
  --tag local/mm3-handoff-r01:zero
```

设置目标机器变量，例如：

```bash
COMMON=(--model-root "$MODEL_ROOT" --draft-root "$DRAFT_ROOT" \
  --gpus "$ALLOCATED_GPUS" --device-group "$RENDER_GID" \
  --lock "$SHARED_GPU_LOCK" --port 6101)
# 若 snapshot 布局需要，另加 --draft-blobs "$DRAFT_BLOBS"。
```

在单独进程／容器中验证 kernel 和真实 Eagle3 C24/C4：

```bash
"$PY" -B tools/run_arm.py --arm r01_compile_prepare \
  --image local/mm3-handoff-r01:zero --output "$RESULTS/r01-smoke" \
  --phase smoke "${COMMON[@]}"
"$PY" -B tools/audit_dispatch.py --arm r01_compile_prepare \
  --run-dir "$RESULTS/r01-smoke"
```

R7/R8 smoke 之前，先运行 `probes/README_zh.md` 中的可移植数值 probe。
其他 E2E 之前，先审查下表所需的正确性／微基准覆盖。
CPU 补丁检查不能代替 GPU 正确性测试。

新测一轮 A0，再独立测每项一轮：

```bash
"$PY" -B tools/run_arm.py --arm a0 --image local/mm3-handoff-a0:zero \
  --output "$RESULTS/a0" "${COMMON[@]}"
"$PY" -B tools/run_arm.py --arm r01_compile_prepare \
  --image local/mm3-handoff-r01:zero --output "$RESULTS/r01" "${COMMON[@]}"
"$PY" -B tools/compare_results.py --baseline "$RESULTS/a0" \
  --candidate "$RESULTS/r01" --output "$RESULTS/r01/comparison.json"
```

对 R2–R8 重复候选构建／验证／E2E，始终复用 `$RESULTS/a0`。
不能复用 smoke 服务来测 E2E。辅助工具会拒绝已存在的结果目录；
不要删除目录来重跑。按用户要求，JIT 和冷启动首波影响属于零预热结果的一部分，
不要丢弃首波，也不要偷偷预热到数字变好为止。

## 正确性及性能归因要求

| 项目 | 归因前必须检查 |
|---|---|
| R1 | eager／compiled target 输出、hidden／aux states 和 cache 写入；四个 rank 均出现融合 QK 前处理 |
| R2 | 实际 M96 及 draft shape 的 BF16 输入／FP32 输出 router；routing／top-k 不变；出现 `_rocm_fp32_router_gemm_kernel` |
| R3 | 三步 draft、positions／lengths／slots／block 边界／已完成请求；出现 fused-step 日志，无 metadata-rebuild fallback |
| R4 | target／draft 贪心结果等价，覆盖相等 logits、vocab padding、fallback；四类分片采样 kernel 均出现 |
| R5 | A0 与 R5 PA 的阈值／空行／page-ID／scale 场景，KV 不变；出现 `mm3_a38_paged_attention_head1_hybrid` |
| R6 | 保留原生 A0 分区／partials，merge 输出一致；两个 Jinja 模板已安装；四个 rank 的 `[1,24,18]` launch 记录证明 cap432 生效 |
| R7 | A0 实际 BM16、BF16 `[96,5,768]` slot intermediate；只改 G1，CK-Tile G2 不变；出现 private TN64 kernel |
| R8 | A0 G1 逐位不变；slot 输入 G2、poison／epoch／counter／IPC 检查；只做一次 TP reduction，无额外 sort／layout kernel，fallback 正确 |

R5/R6 旧的 donor 准备测试不是可直接在目标机器运行的独立测试。
旧 GPU fixtures 需要适配已安装的 R5/R6，且 R6 的 reference 必须是原生 A0，
**不能是 A38**。不要宣称补丁包已完成这些 GPU 数值验证。
旧 fixtures 不可用时，基于当前实现编写隔离的 kernel 微基准。
必须统计完整组件成本，不能只截取缩短后的 kernel 片段。

R7/R8 包含**新的** A0 布局适配，并非原封不动的 A48/A37 kernel。
GPU 数值验证是必需项，目前尚未完成。
R7 私有 activation helper 有意与全局 A0 AITER helper 不同，
不要替换全局 helper。R8 包含两个精确版本的原生库；
不能用另一套 PyTorch ABI 重编译后仍声称旧哈希有效。

## 已有证据与最终交付

历史 A0 吞吐为 2310.4520822142695 tok/s，使用 48 次 HTTP 预热，
启动时选择 GPU 0–3。早期独立 R1/R2/R3 有 19 对 37 项 capture 列表不一致。
修正列表后的 R1/R2 用了零 HTTP 预热，而历史 A0 有预热，
所以表面上的 −6.53%／+0.91% **不是独立优化收益**。
不要把这些历史数字当作目标机器的 baseline 或验收目标。

本次交接只做打包／CPU 检查，没有新做 E2E 或 kernel 速度测试。
原累计 A48 未改动。

最终输出一张表：目标机器 A0 和每项的 image／patch 哈希、正确性及实际调度状态、
微基准结果、输出吞吐、mean TPOT、P99 TTFT／ITL／E2EL，
以及相对共用新 A0 的变化。保留失败／持平／负向项，链接原始产物，
区分数值差异、因果结论和统计显著性。不要相加八项百分比，
也不要自动升级为累计优化分支。
