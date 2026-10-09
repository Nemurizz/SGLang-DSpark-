# DSpark 优化阶段总结：FP8 量化与失败路线归档

日期：2026-10-09（Asia/Shanghai）

这次更新从 GitHub 上一次已上传的 h66 协议提交 `72b1019` 继续，重点不是重写 DSpark，也不是替代 CUTLASS，而是先用真实任务和稳定态 A/B 判断量化路线是否值得保留。

## 为什么选择原生 W8A8 FP8

前一阶段对 NVFP4 做了两条独立路线：原生 CUTLASS 和原生 cuDNN。两条路线都保留了完整启动、长输出和服务安全证据，但在真实长数学输出的固定时限内没有完成可靠的端到端资格验证，因此没有提高超时上限、删题重试或宣称收益。

受限 nsys trace 显示 NVFP4 路线的 GPU kernel busy 只占捕获区约 16.19%，同时存在显著的 host dispatch/量化开销；这说明继续手写 GEMM 不能直接解决测到的问题。于是选择 SGLang 已有的 Blackwell W8A8 FP8 路径作为保守替代：模型仍加载原始 BF16 parent，160 个 target fused linear 使用 per-output-channel FP8 权重和动态 per-token FP8 activation，head/embedding、draft、KV 和固定 FA4 保持 BF16。

这条路线没有下载新模型、没有升级依赖、没有写普通 Triton 或替代 CUTLASS 的 GEMM。通过 `USE_TRITON_W8A8_FP8_KERNEL=0` 和 `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0` 固定到已安装的 native `sgl_kernel.gemm.fp8_scaled_mm`，并在运行时校验权重 dtype、scale 形状、层数量、维度对齐和未量化模块。

## 怎样验证

1. 先做实际 Qwen 形状 M=1/8/32/64 的 native backend smoke。
2. 生成量化路线自己的 AR token oracle，再要求 DSpark 在串行、1/4/8 并发和两轮重复中逐 token 一致；DSpark 额外逐项比较五个 acceptance 字段。
3. 在完全相同的 384 个固定题目上分别跑 BF16 和 FP8，固定 tokenizer、prompt、seed、停止条件、代码沙箱和 grader。用户约束为每个任务最多下降 1 个百分点。
4. 质量通过后再做未开启 nsys 的稳定态 ABBA：2K prompt、128 output、并发 1、两边 `max_total_tokens=16384`，graph/cache/adaptive draft/KV quantization 全关闭；先通过 BF16/BF16 噪声和 AB/BA 漂移准入，再比较两个顺序。

## 结果

### 质量

| 任务 | BF16 | W8A8 FP8 | 变化 |
|---|---:|---:|---:|
| GSM8K | 113/128 | 119/128 | +4.69pp |
| 标量 MATH | 25/64 | 26/64 | +1.56pp |
| ARC generative | 118/128 | 121/128 | +2.34pp |
| MBPP | 52/64 | 54/64 | +3.13pp |

四项均未下降，固定子集质量门槛通过。MATH 是固定 4096 输出上限的标量子集，不是完整 MATH500 leaderboard；BF16 有 21 个截断、FP8 有 20 个截断。这个结果证明的是本固定 protocol 内的 observed quality，不是普适统计非劣性。

### 稳定态 ABBA

| 顺序 | BF16 E2E mean/p95 (ms) | FP8 E2E mean/p95 (ms) | mean gain |
|---|---:|---:|---:|
| AB | 2717.795 / 2723.649 | 1864.534 / 1870.851 | 31.40% |
| BA | 2716.766 / 2719.389 | 1856.716 / 1863.705 | 31.66% |

TTFT p95 比例分别为 0.6512 和 0.6533，TPOT p95 比例分别为 0.6906 和 0.6887，E2E p95 比例分别为 0.6869 和 0.6853，均没有回退。峰值显存从 36177 MiB 降到 24129 MiB，约降低 33.3%。

FP8 acceptance 为 `6016 / 14784 = 40.70%`，BF16 为 `5760 / 17472 = 32.97%`；同一配置内的 acceptance 在 sentinel 和 ABBA 检查中一致。

## 如何解释收益

本次收益属于量化后的 target linear compute 和权重/工作集缩减带来的收益。两边 KV 容量被显式固定，避免把自动增大的缓存空间误算为算力收益；计时也没有启用 paired cache，所以没有把 cache hit 当作 FP8 收益。

这不是对所有 DSpark workload 的承诺。高并发只在 correctness sentinel 中验证过，还没有做高并发性能 ABBA；本次 ABBA 的 cache gauges、shadow hit、paired hit 和 partial recovery 都是空值，不能用于 paired-cache 结论。

## 后续最值得做的事

1. 把已通过的 FP8 recipe 单独做 4/8 并发性能 ABBA，仍保持 equal-KV-capacity、p95 guard 和 token/acceptance guard；如果高并发不稳定，不放宽 guard。
2. 另开 paired-cache 专用 A/B，比较 BF16 与 FP8 时固定 prompt 复用序列，单独记录 shadow hit、paired hit length、partial recovery、fallback、TTFT/TPOT/p95 和显存；不要把它与本次 FP8 compute gain 合并成一个结论。
3. 用 nsys 继续确认真实服务中 FP8 weight GEMM、FA4、KV write、compact/scan 和 scheduler metadata 的占比；不写替代 CUTLASS GEMM。
4. 只有当质量或高并发结果暴露瓶颈时，再考虑 MLP-only FP8/BF16 attention、FP8 block-scale 或经过独立 calibration 的混合精度；每条路线都必须新 oracle、新质量门槛和新 ABBA。
5. ragged graph 仍保持 exact geometry guard；graph catalog 或常见 geometry 预捕获必须逐 geometry 做 token-level correctness 和稳定态 A/B。

## 可复现与 Git 状态

- 分支：`exp/quant-w8a8-fp8-native-target-bf16-draft`
- 结果提交：`7793a91`
- 结果 tag：`dspark-native-w8a8-fp8-quality-abba-passed-20261009`
- serving checkout 的 frozen source hash、`main`、未跟踪 baseline 和原始参考均未改变。
- 完整 h66 原始证据包含 raw requests、token ids、server logs、GPU telemetry、quality report、ABBA summaries 和 manifests。
