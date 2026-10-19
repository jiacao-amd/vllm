# 已接受 N18：MiniMax-M3 跨机器验证

[English](README.md) | **中文**

本包移交的是已接受的 **28文件 vLLM 覆盖层**，不是历史上八个独立的 A0 消融实验。
累计组合为 **N02 + N05 + N06 + N07（含 N03 采样）+ N12 + N16 + N18**。
N16 替换 N14 的 G1 配置。**AITER 保持不变。** 优化目标仍处于暂停状态；
本次移交不声称已经达到9%收益。

## 给接手代理的提示词

> 在另一台兼容机器上验证这个精确的已接受 N18 分支。
> 执行前阅读两个清单及冻结协议。以固定摘要的官方镜像构建源码覆盖层，
> 不重新编译 vLLM，不替换 AITER、内核、模型权重或基准测试依赖。
> 两组实验使用相同的四张 gfx950 GPU，独占这些 GPU，并保持引擎／客户端参数一致。
> 每组只用冻结的原生 dummy 工作负载准备编译缓存。标准 Eagle3 C24/C4 冒烟测试
> 必须使用独立进程和各自空的私有缓存；检查四个 rank 的轨迹，确认保留实现实际执行，
> 不能只看文本输出成功。每组先通过自身门禁，再各运行一次 N0 和 N18 正式 E2E：
> 240个请求、C24、8192/1024，且 **`--num-warmups 0`**。
> 以这台机器本次测得的 N0 为分母。保存所有日志、失败、源码／镜像哈希、GPU UUID、
> 缓存审计和原始结果。不得用冒烟或 E2E 结果反向填充缓存，不扣除 JIT／停顿，
> 不筛选有利请求，不因结果为负而重跑。报告吞吐量及全部已保存的延迟分位数；
> 身份、正确性或分派门禁失败即停止。不要恢复优化或修改代码。

## 固定环境与历史证据

| 项目 | 固定版本／协议 |
| --- | --- |
| 官方基底 | `vllm/vllm-openai-rocm@sha256:c6cb4ab021e31691f63a5757278153fbe728a71c722651a4549ddbb5ac3067ab` |
| vLLM 基底 commit | `1b2a666391f2a24228a1436b64d547c34cc6b6ce` |
| 运行时 | Python `3.12.15`；vLLM `0.32.0rc1.dev369+g1b2a66639.rocm72`；Torch `2.13.0+git733fca1`；HIP `7.2.53211` |
| 内核依赖 | Triton `3.8.0+git669b31ac`；AITER `0.1.25`；FlyDSL `0.3.4.1` |
| 硬件 | 四张 MI355X／`gfx950` GPU，TP4；记录物理 UUID，两组选择完全一致 |
| 来源机器（历史） | `smci355-ccs-aus-m15-17`，GPU4–7，UID:GID13807:13807，附加设备组110；远端不要求同名用户／同组号 |
| 模型 | 主模型快照 `229447a2d64149ffb45e454ebeb74b7a9a45ce7d`；草稿模型快照 `96692486b5fd38ebf8fd2a5f6bb53427d30819a8` |
| 精度 | 原始 MXFP4 模型、BF16 activation、原始 FP32 router 权重、FP8 主／索引 KV；不转换精度 |
| 引擎 | Block128、显存利用率0.9、启用 chunked prefill／前缀缓存、token budget32768、max sequences48；HTTP 并发为24 |
| 图捕获 | `FULL_DECODE_ONLY`，严格使用 [protocol.py](protocol.py) 中的 **37个尺寸**；只设 max128 并不等价 |
| 正式工作负载 | 240个请求、固定8192输入／1024输出 token、C24、seed0、chat template、ignoreEOS、stream interval20；Eagle3 K3、合成接受长度2.78、thinking enabled |
| 独立冒烟 | 原始标准（非合成）Eagle3 C24/C4 正确性检查，thinking disabled；不作为计时或编译缓存准备工作负载 |
| 协议 ID | `official-nightly-tp4-c24-8k1k-http0-dummy-compile-v1` |

下面是**来源机器上的历史单次结果**，不是远端基线：

| 实验组 | 输出 tok/s | 变化 |
| --- | ---: | --- |
| 原始官方 N0 | 2414.2174046171085 | 仅作为来源机器的分母 |
| 已接受 N18 | 2546.872092705771 | 相对 N0 +5.49472834696%；相对 N16 +0.6675924329% |

N18 相对 N16 的延迟变化有好有坏：mean TTFT **+3.5898%**、p99 ITL **+2.3961%**；
mean TPOT **−1.1724%**、p99 E2EL **−1.2353%**、p99.9 ITL **−3.0054%**。
P99 ITL **相对 N0 +0.7218%**。保留观察到的两类运行时 JIT 警告及其全部耗时，
不扣除时间、不声称统计显著性。N06 每 GPU 额外保留588MiB 的 packed head 和回退权重。
N18 的 BF16 分项能精确重建已检查的 router 权重；这**不表示**
BF16x3 点积与原生 FP32 点积逐 bit 相同。

## 旧 R 主题与当前 N 的对应关系

这是主题映射，不代表逐字节移植，也不能把收益相加。

| 旧主题 | 当前已接受 nightly 中的状态 |
| --- | --- |
| R1／A1 attention preparation | 原生 N0 已有带 guard 的 norm／RoPE／cache 融合准备。旧 compile 接入并未完整复现，也不重复计收益。 |
| R2／A4 router | N01 独立 E2E 仍被拒绝；其支持范围扩展通过已接受的 N12 保留。N18 另加精确 M96 BF16x3 router，原权重及其他 M 的回退保留。 |
| R3／A5 草稿 metadata | N07 按当前 metadata-update ABI 实现草稿循环融合，不覆盖旧版源码。 |
| R4／A33 目标／草稿采样 | 独立 N03 仍被拒绝；采样实现保留在已接受的 N07 组合中。 |
| R5／A38 稀疏 PA | N05 对实际单 KV head 路径做 partition，保留原生 merge／回退。 |
| R6／A39+A41 merge／score | N02 保留 score cap 的剩余优化。旧 grouped-merge 补丁未另行安装或计收益。 |
| R7／A48 G1 | N14 建立当前布局集成；N16 将精确 M96 改为 TN64/TK128/w2，替换 N14 的 TK256，并非旧 A48 payload。 |
| R8／A37 G2+TP | **未保留。** N13 fusion／transport／publication 实验未获准入，仍使用原生 G2/TP。 |

N06 packed draft head 是额外的已接受 nightly 主题，不属于旧八组消融。
不要在本包之上再应用旧 R 补丁。

## 包内容与构建

[accepted_manifest.json](accepted_manifest.json) 固定28个最终源码哈希和 N18 必需开关。
[native_manifest.json](native_manifest.json) 固定官方源码／依赖清单。
`install.py` 校验原生前像及已接受覆盖层；Dockerfile 必须以**仓库根目录**为构建上下文。
冻结的 `protocol.py`、`prepare_cache.py`、`compile_extension.py`、`model_check.py`、
`reference_run_e2e.py` 及 `client/*.py` 保留原工作负载。可移植入口是 `run_inside.py`，
**不要**执行 `reference_run_e2e.py` 内原机器的 host launcher。
已接受源码有历史 GPU 证据；新打包／wrapper 仅有 CPU／构建检查，
不代表又完成一次 GPU 测量或远端验证。

**这是复现分支，不是可直接合并的上游 PR。** 只读的拼写、root-import、
forbidden-boolean、配置及路径检查通过，但上游 mypy 报告 **6个文件共46个错误**
（包括动态 FlyDSL 类／属性及 Optional 类型收窄）。forbidden-import 检查还标记了
精确保留的 `gpu_pack.py` 中直接导入 Triton 的写法。完整上游静态检查**并未全部通过**；
移交期间没有进行类型、导入或格式清理。

新 installer／wrapper 的 **7项 CPU 测试、Ruff 及完整584源码镜像审计通过**，
不代表所有 pre-commit hook 均通过。计划中的移交提交将明确记录跳过的已知 hook，
以保留实际测量过的 payload 字节；上游清理另行处理。本次移交未重新运行 GPU 测试。

以下命令供接收方操作；编写这些说明本身未执行 GPU／Docker。
在已检出的仓库根目录运行：

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

保存构建日志及解析后的镜像 ID。重建的 N18 镜像可以有新的 ID；
必须匹配本包源码哈希，不要求与来源机器的历史镜像 ID 相同。

## 模型挂载、权限与执行

将下面变量设为 host 绝对路径。`MAIN_CACHE` 和 `DRAFT_CACHE` 是各模型的
Hugging Face 缓存根目录，包含 `snapshots/` **及所有被引用的 blobs**。
快照文件可能是符号链接，仅复制 `snapshots/<hash>` 不够。
如果链接指向共享父目录下的 `blobs/`，还须把对应目录挂载到正确的容器路径。
也可提供已完全解引用、实体化的快照目录，挂载到以下精确路径；
不得静默改用其他 revision：

```text
/models/main/snapshots/229447a2d64149ffb45e454ebeb74b7a9a45ce7d
/models/draft/snapshots/96692486b5fd38ebf8fd2a5f6bb53427d30819a8
```

示例假定所有链接都能在各自挂载的模型缓存根目录内解析。
加载模型前先确认这一点。指定的非 root UID/GID 须能读取模型／包，
并能写入结果及缓存目录。原运行使用 UID:GID13807:13807、设备组110；
远端应使用当地实际 render-device 组，不能默认也是110。

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

全程持有该机器约定的 GPU 锁，并确保没有竞争任务使用这四张卡；各阶段串行执行。
下列函数每阶段启动新的非 root 容器和临时 system-site-packages venv，不安装依赖。
`/suite`、`/client` 只读，`/results` 必须是全新目录，只有该阶段私有 `/cache` 可写。

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

`run_inside.py` 应用／检查冻结环境及各组开关；不要给 N0 启用 N18 开关，
不要改变已接受配置。**每个阶段**都必须提供由 Docker inspect 得到的
`MM3_HANDOFF_IMAGE_ID`，并运行同一个已解析 ID。保持端口6101空闲。
控制台日志位于挂载结果目录之外，`status.json` 记录完成状态。
临时 uv 缓存放在 `/cache` 外，避免引导过程污染编译模板；
空缓存检查前不要预先创建 `/cache/home`。

### 准备 → 冒烟 → 每组一次正式 E2E

先运行 **N0 准备 → N0 冒烟 → N0 E2E**，再对 N18 重复同一顺序。
**任一阶段失败即停止。** 准备及冒烟各自从独立空缓存开始；
只有正式 E2E 使用对应组准备模板的副本。

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

准备阶段须通过 `compile_ready.json`，冒烟须通过 C24/C4 `model_check.json`
及所有 rank 的分派检查。准备产物中的 `cache_fingerprint.json` 固定缓存字节，
在 E2E 前校验。模板必须保持准备结束时的原样，不得把冒烟产生的文件复制回去。
正常编译／原生图预热与 HTTP 预热是两回事。

每次正式 E2E 须完成240/240请求、1966080输入和245760输出 token，
使用精确37个捕获尺寸，并且**零 HTTP 预热**。`/prepared` 是对应组的只读编译准备产物。
两组不共享缓存，也不能将冒烟／E2E 缓存作为新模板。保留失败，
先调查，再决定是否另行授权新尝试。
若接收机器已按这个精确包／协议完成 N0，应保留其结果，
只运行尚未完成的 N18 阶段；不能因为候选准备失败而重测 N0。

仅用这台机器的两次测量，报告
`100 * (remote_N18_output_tok_s / remote_N0_output_tok_s - 1)`。
同时提交 mean／p90／p99／p99.9 TTFT、TPOT、ITL、E2EL、请求／token 总数、
镜像／源码 pin、GPU UUID、缓存审计及原始日志。
stream interval20 时，客户端 ITL 是流式 chunk 间隔，不是逐 token 内核延迟。
冒烟成功或组件更快不等于 E2E 接受；远端单次成对验证也不代表统计显著性。
