# baseline 并发差异：已定位首个变化算子，尚未修复

收益矩阵没有启动，paired cache 从未开启。本轮只定位原 baseline 的数值差异，并验证一个默认关闭的现有精度配置候选；没有新 GEMM kernel、graph 或量化实验。

## 原 baseline 的证据

配置为原 `a61e0bb`、deterministic OFF、graph 全 OFF、greedy、32-token 输出，实际 cache reuse 为零。复现固定 `p128_c4 / variant 0`：原冷串行第 3 个输出 token 为 `34334`，并发为 `16555`。串行重复的所有保存张量逐位一致。

读取的是首个已提交 anchor，位置 128、token 576；并非把未来 draft token 当成已提交历史。该 anchor 本身预测第 2 个输出 token，不能声称直接保存了第 3 个分歧 token 的完整计算。

| 第一层边界 | B1 与 B4 的不同元素数 | 最大绝对差 |
| --- | ---: | ---: |
| 历史 prefill K / V | 0 / 0 | 0 |
| input norm 输入 / 输出 | 0 / 0 | 0 |
| QKV 投影输入 | 0 / 5120 | 0 |
| QKV 投影输出 | 3022 / 7168 | 0.00390625 |
| Q / 当前 K / 当前 V | 2338 / 469 / 505 | 0.0625 / 0.0625 / 0.0009765625 |

KV 写入逐位吻合当前 K/V 输入。后续层和 logits 的差异随传播扩大。由此可以定位**最早观察到的差异是第一层 QKV 投影**，不能归咎于 paired sidecar、KV 写错或只改 attention planner 来解决。

## 精确复算与候选

从原 checkpoint 读取 TP1 的 Q/K/V 权重，使用 SHA 校验后的相同投影输入，以实际 M=8 和 M=32 复算 `torch.nn.functional.linear`。两种原输出均逐位复现，差异同为 3022 个元素。这个复算无计时，不是性能 microbenchmark。

关闭现有 `allow_bf16_reduced_precision_reduction` 后，M=8/M=32 的输出差异降至 3 个元素，但仍非逐位一致，并且原 B1 输出改变 2553 个元素。它只能作为正确性候选，不能直接宣布修复。

候选为 `SGLANG_DSPARK_BF16_FULL_REDUCTION=1`，默认 OFF，严格限定已检查的 Torch 2.13/cu130、SM120、BF16、TP1、无量化、graph OFF、deterministic OFF 和 torch 路径。没有改动 cache guard 或 graph geometry guard。

端到端结果：

- `p128_c1`：原冷串行参考与 3 轮 serial/concurrent 的 token、5 项 acceptance 字段均通过。
- 接着检查 `p128_c4 / variant 0` 的原冷串行参考时，**第 22 个 token 已改变**：原 `2169`，候选 `8131`。
- 保存的 acceptance 同时变化：correct drafts `22 → 20`，proposed drafts `70 → 77`，verify 次数 `10 → 11`。门槛先报告 token 失败，并未忽略 acceptance。
- 立即停止服务，尚未进入该 cell 的并发测试，未启动 paired cache 或性能矩阵。
- 候选因此拒绝；不能重建候选 oracle 来掩盖这个失败。

候选最终净改动（相对 `a61e0bb`）只有 4 个文件、100 行新增；保留在 `exp/cache-bf16-full-reduction@a746a1f3ff6783c53a9fa8bd97809b1e33365f9d`，拒绝 tag 为 `dspark-bf16-reduction-rejected-20261007`。不要单独使用早期 `aaac87b`：它曾误纳入原未跟踪 runner，随后已纠正。

## 已回退与理论边界

服务已停止，GPU 0 MiB / 0%。工作 checkout 已回到 `exp/reuse-compact-index@a61e0bb`，与回退 tag `dspark-cache-bf16-reduction-parent-20261007` 一致。原始参考不变，main 未改、没有 push。

原 `model_runner.py` 是未跟踪基线文件。加入它的候选代码和跟踪状态已撤销，文件仍存在；当前内容与保存副本 SHA256 都是 `9d09f6d78d01ef3c7b58419cfae4367c80aec1a6dcc94f9af551cb90da32a1b9`。未跟踪文件数恢复为原 8357。

当前旧 BF16 路线对矩阵 M 形状敏感；并发和 prefix reuse 都会改变实际 M。按请求拆分已有计算可能保留部分旧 B1 舍入，但会增加算子启动次数；仅修 verify 也不能保证 draft、prefill、partial recovery 的完整一致性。未经证明，不应为它盲目跑收益矩阵。

下一步需要明确 baseline 合同：若必须复现所有旧 B1 token/acceptance，只能继续研究保留原算术分区的框架调度候选，收益仍未证明；若允许建立新的独立数值稳定配置，则必须保留旧参考、单独记录配置，并重新通过独立 target 对照、1/4/8 并发及冷/热/partial cache 正确性门槛，才做同配置 A/B。后者不是简单放开 deterministic 的 cache guard。

本报告配套的 `attention_comparison.json`、`projection_comparison.json`、`qkv_rounding_replay.json` 和 `precision_probe_status.json` 给出原始差异与停止条件。诊断源代码、服务日志、张量、数据集和补丁另存证据包。
