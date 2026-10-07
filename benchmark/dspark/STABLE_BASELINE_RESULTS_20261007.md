# 独立数值 baseline：32-token 全矩阵初筛通过

本轮按授权建立独立算术配置，保留旧参考；不要求复现旧 BF16 舍入结果。
结论限于固定数据集、32-token 输出、三轮串行/并发检查。尚不是生产稳定性、长输出或性能收益证明。

## 已验证结果

`128/512/2048/8192 × 1/4/8 并发` 共 12 格：

| 检查组 | 保存请求 | 结果 |
| --- | ---: | --- |
| 独立普通 target，禁止 prefix reuse | 416 | 冷串行、重复串行、并发逐 token 一致 |
| DSpark full-prefill，实际 cache reuse=0 | 416 | 匹配独立 target；五项 acceptance metadata 完全一致 |
| DSpark paired-cache | 416 | 冷请求、不同 suffix 的 prime、三轮热串行/并发均匹配冷对照；每条热请求实际 cached_tokens>0 |

总计 1,248 条已保存请求，不含每组 4 条热身。另有 `128 × 1/4/8` 聚焦复验 312 条通过。
本轮相关单元测试最终24项全部通过、无跳过；原ragged10项与backend capability2项也通过。
五项严格比较：verify 次数、correct/proposed drafts、acceptance/correct-draft histograms。
冻结 52 条完整 DSpark 冷参考，phase 已规范为 `reference`，可由现有 frozen-reference loader 读取。

新参考 SHA256：
`05c5c3b0397728ace413ac49783f18d845f8118fcb0a9d6dd853a6efd185427f9`

数值 profile ID：`dspark-stable-fa4-fixed64-v2-5d2e701bc65394216f57e47b`。

312条热串行/并发请求的 `cached_tokens` 按prompt长度分别为127、511、2047、8191，三种并发下相同。这是逐请求实际target复用长度；sidecar的target/paired匹配计数另保存在cells.jsonl中，不能用多次match调用数冒充请求级hit rate。

## 方案与因果证据

配置使用现有框架：deterministic/batch-invariant ops、torch BF16 GEMM、FA4 target/draft、BF16 KV、TP=1、page=1、num_splits=decode_num_splits=1、PyTorch greedy，所有 graph 关闭；动态 draft length 和 CUDA compact/scan 关闭。原 SPS 固定。

第一版“只开启 deterministic FA4”没有通过：普通 target 的聚焦检查通过，但 DSpark 第一个冷请求的第三个输出 token 为16555，普通 target 为34334。立即停测、回到旧主实验检查点，未进入 paired。

只读快照定位到同一已提交位置 `position128/token576`：输入、QKV 投影、Q/K/V、历史 KV 和当前 KV 写入逐位相同；layer0 attention 输出113/5120个 BF16 元素不同，最大绝对差0.0001220703125，后续层差异放大。每条路径重复两次均逐位稳定。该位置预测第二个输出 token，不能把它误说成第三个 token 分歧位置的直接快照。

保存 Q/K/V 的离线重放逐位复现两条实测 attention 输出。现有 SM120 host 自动选择 decode64×64、verify64×128；用现有64×64或128×64接口固定 N=64 后，这个保存位置的 Q1/Q8 输出逐位一致。随后上述完整端到端矩阵验证了候选，不能用离线重放本身宣称端到端正确或提速。

新候选通过显式 opt-in hook factory，在首个 FA4 编译/加载前，将该进程的现有 SM120 host selector 固定为64×64；没有编写或修改 attention/GEMM kernel。这个 factory **会改变数值调度配置，不是只读 observer**。普通/default 服务不会加载它；worker退出后进程级配置消失。只支持已核对的 Qwen3 40Q/8KV、HD128几何；保留 target 的因果与 block draft 的非因果语义，拒绝 local/bias、冲突 tile、已有编译条目及 graph 开启。

启动适配错误也全部保留：h28误把 JIT 包装对象真值当作缓存非空；h29误拒绝原本非因果的 block draft。两轮均停止并回到旧分支，不包含合格的 DSpark 测试结果。修正后 h30/h31从头重跑，未跳过正确性检查。

## 保存与回退

- 独立分支：`exp/numerically-stable-fa4-fixed64`。
- 已验证代码提交：`4e3a7bcbb874b021ddbe695ab1fbf8dd2eeb0a25`。
- 范围限定的通过 tag：`dspark-stable-fa4-fixed64-matrix32-passed-20261007`。
- 原主实验：`exp/reuse-compact-index@a61e0bb`；原始 main 未修改。
- 第一版失败配置：`exp/numerically-stable-fa4-baseline@9a9aa71`，失败 tag保留。
- 本轮新增/修改仅在 benchmark 目录。分支还继承先前默认关闭的 FlashInfer reference-plan 实验代码，不能将整个分支相对a61的差异都说成只有本轮配置。

旧52条参考 SHA仍为 `cee34131dd7431170f09b431d9b9147220d9b09f15a9aeef861fc16d81795de6`；h21旧请求/数据集与13个核对的核心源码哈希未变。8357个原未跟踪文件保留，原未跟踪 model_runner 的 SHA仍为 `9d09f6d78d01ef3c7b58419cfae4367c80aec1a6dcc94f9af551cb90da32a1b9`。

所有本轮测试服务均已停止，结束时 GPU为0MiB/0%。没有 push、force push、合并main或删除 baseline 文件。

## 尚未证明与下一步

1. 32-token初筛不能替代128/512-token输出和持出 prompt 的验证；新长度必须建立、验证自己的独立 target/DSpark 冷参考。
2. partial/full recovery 与 eviction 的本轮计划事件为0：这些压力路径未覆盖，不是“恢复能力已证明”。sidecar计数是计划，不是 worker 实际执行次数；真实 fallback 仍未独立计量。
3. 正确的 target-only + draft recovery 对照尚未建立；不能宣称 paired 优于它。
4. TTFT/TPOT/p95/显存的正式收益 **未证明**。正确性 gate 的固定 full→paired顺序、32-token输出和样本量不能充当稳定态 A/B。新 baseline 的固定 tile 与 batch-invariant 算术本身也未与旧框架做有效性能比较。
5. 下一阶段先做长输出、partial recovery/eviction正确性与实际工作量 instrumentation，再按A/B、B/A平衡轮次测固定12格；只改变 cache treatment，使用同一新数值profile。不能把新算术配置的成本/收益混入 paired-cache收益。

此前协议限定原配置 deterministic OFF，是原 baseline 的历史条件；本轮授权后的新profile在两条A/B臂均使用deterministic ON且FA4，不能跨profile比较或放宽旧FlashInfer cache guard。

GEMM新kernel、graph和量化实验均未开展。

远端完整结果目录：`/root/autodl-tmp/results/dspark-stable-fa4-fixed64-matrix-20261007-h31`。
复现32-token检查：

```sh
PYTHONPATH=python:benchmark/dspark /root/conda-envs/sglang/bin/python \
  benchmark/dspark/stable_baseline_gate.py \
  --template /root/autodl-tmp/results/dspark-cache-compatible-original-baseline-20261007-h21 \
  --expected-head 4e3a7bcbb874b021ddbe695ab1fbf8dd2eeb0a25 \
  --fixed-fa4-tile --cells all --with-paired --output NEW_RESULT_DIRECTORY
```

必须位于已验证commit、tracked clean且输出目录不存在；baseline策略不由普通服务启动自动启用。
