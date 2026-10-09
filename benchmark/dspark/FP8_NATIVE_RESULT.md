# 原生 W8A8 FP8 结果（h66）

日期：2026-10-09（Asia/Shanghai）

分支：`exp/quant-w8a8-fp8-native-target-bf16-draft`，HEAD `72b1019`。
这是独立工具/验证分支；没有修改 `main` 或原始 serving checkout，也没有写自定义 GEMM。

## 质量

固定数据集 `dspark-quality-20261009-h64-v1`，384 个题目，逐题配对，评分规则和提示词不变。用户门槛为每项最多下降 1 个百分点。

| 任务 | BF16 | W8A8 FP8 | 变化 |
|---|---:|---:|---:|
| GSM8K | 113/128 (88.28%) | 119/128 (92.97%) | +4.69pp |
| 标量 MATH | 25/64 (39.06%) | 26/64 (40.63%) | +1.56pp |
| ARC generative | 118/128 (92.19%) | 121/128 (94.53%) | +2.34pp |
| MBPP | 52/64 (81.25%) | 54/64 (84.38%) | +3.13pp |

固定子集质量门槛：`passed_for_fixed_subsets_only`。这不是完整榜单或普适非劣性结论；MATH 使用固定 4096 输出上限（BF16 21 个截断、FP8 20 个截断）。

普通 AR 和 DSpark 各 56 个 sentinel 检查全部通过，覆盖串行及 1/4/8 并发两轮；同一配置内 token 逐条一致，DSpark 五项 acceptance 也一致。

## 稳定态端到端 ABBA

固定单元：2K prompt、1 并发、128 输出 token、两边 `max_total_tokens=16384`，CUDA graph/cache/adaptive draft/KV quantization 关闭。BF16 噪声和 AB/BA 漂移均通过准入。

| 臂 | TTFT mean/p95 (ms) | TPOT mean/p95 (ms) | E2E mean/p95 (ms) | 峰值显存 |
|---|---:|---:|---:|---:|
| BF16 (AB) | 257.678 / 259.305 | 19.370 / 19.408 | 2717.795 / 2723.649 | 36177 MiB |
| FP8 (AB) | 167.864 / 168.861 | 13.358 / 13.402 | 1864.534 / 1870.851 | 24129 MiB |
| FP8 (BA) | 167.900 / 168.800 | 13.296 / 13.352 | 1856.716 / 1863.705 | 24129 MiB |
| BF16 (BA) | 257.242 / 258.353 | 19.365 / 19.388 | 2716.766 / 2719.389 | 36177 MiB |

- AB E2E mean gain：31.395%，p95 ratio 0.6869，p95 未回退。
- BA E2E mean gain：31.657%，p95 ratio 0.6853，p95 未回退。
- 候选 acceptance：6016 correct drafts / 14784 proposed，accept rate 40.70%，completion tokens/verify 3.8788。
- BF16 acceptance：5760 / 17472，accept rate 32.97%，completion tokens/verify 3.2821。

因此，在这个明确限定的单元上，质量和 >1% 端到端收益均已证明。不能外推到高并发、不同 prompt/output 长度或带 paired-cache 的收益；本计时单元没有缓存命中，cache gauges 为空，shadow/paired hit/partial recovery 不应从该结果推断。

完整原始证据（请求、输出、服务日志、遥测、manifest、质量报告、ABBA 结果）：
[h66-v2 归档](../../outputs/dspark-quant-fp8/dspark-quant-fp8-quality-20261009-h66-v2.tar.gz)

归档 SHA-256：`6c794c1d1e3901757d2333810ea1989f576c44b2da85da50ac45b192a7d6fae9`。
