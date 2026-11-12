# NVFP4 小 M kernel / scale-layout 冒烟实验（2026-11-13）

## 结论

本轮尝试了几种针对 DSpark 小 M verify 的 NVFP4 优化方向。只有“对小 M 关闭 PDL”能够在独立 GEMM microbenchmark 中得到小幅收益，但在服务级 AB/BA 冒烟中没有稳定方向，因此没有扩展到完整矩阵，也没有把它作为默认优化提交。

当前结论仍是：现有 NVFP4 路径需要更深的 kernel 级改造，单独调整 PDL 或 scale layout 不足以抵消 FP4 target 在 DSpark 下的整体开销。

## 共同协议

服务冒烟固定为 `p512/c4`，输出 128，8 个 measured requests、1 个 warmup batch，paired prefix cache、DSpark block7、FA4 fixed64、CUDA Graph/compact scan/adaptive draft/KV quantization 关闭。执行顺序为 BF16 A → NVFP4 A → NVFP4 B → BF16 B，分别记录 AB/BA。target/draft、SPS 表、请求数据和服务参数与此前 NVFP4 矩阵保持一致。

独立 GEMM 使用 Qwen3-14B 线性层的 `K=5120,N=17408`，扫描 `M=1/4/8/16/32/64`，比较 FlashInfer CUTLASS NVFP4 的预量化执行和量化加 GEMM 总时间。正确性比较使用同一输入、同一权重和同一 scale。

## 候选一：小 M 关闭 PDL

实验开关为 `SGLANG_NVFP4_SMALL_M_DISABLE_PDL=1`，阈值为 `M<=16`。它同时传给 FlashInfer FP4 quantize 和 GEMM；未设置开关时保持原始行为。PDL 是可编程依赖启动机制，目标是减少小 M 下的固定启动成本。

独立 microbenchmark 的量化加 GEMM总时间如下：

| M | PDL 开启 | M≤16 策略 | 变化 |
|---:|---:|---:|---:|
| 1  | 0.1877 ms | 0.1805 ms | −3.9% |
| 4  | 0.1862 ms | 0.1794 ms | −3.7% |
| 8  | 0.1791 ms | 0.1749 ms | −2.4% |
| 16 | 0.1741 ms | 0.1700 ms | −2.4% |
| 32 | 0.1691 ms | 0.1869 ms | 方向不稳定 |
| 64 | 0.1500 ms | 0.1540 ms | 变慢 |

PDL 开启和关闭在 `M=1/4/8/16/32` 的输出逐元素完全相同。这个结果说明 PDL 对小 M 的 kernel 微路径有可测影响，但收益只有几个百分点，且存在明确的 M 阈值。

服务级结果没有复现出稳定收益：

| 重复 | 配置 | AB E2E gain | BA E2E gain |
|---|---|---:|---:|
| 1 | 原始 PDL | −44.56% | −42.92% |
| 1 | M≤16 关闭 PDL | −43.26% | −44.84% |
| 2 | 原始 PDL | −39.05% | −41.41% |
| 2 | M≤16 关闭 PDL | −44.61% | −40.35% |

两轮中 AB 和 BA 的变化方向相反，说明这点 microbenchmark 收益被服务级请求轨迹、排队和其他 kernel 波动淹没。两种配置的 acceptance 都是 0.3067，NVFP4 峰值显存都是 20979 MiB，因此不能把差异归因于质量或显存变化。

**处理决定：** 不跑完整矩阵，不作为默认行为提交。实验开关已从活动源码中撤回；它可以作为后续针对固定 M 分布的 profiling 候选保留。

## 候选二：小 M 使用未 swizzle 的 scale layout

未 swizzle 的 scale 在 M=1/4 的 isolated timing 看起来更快，但 CUTLASS `mm_fp4` 没有相应的 layout 参数，直接传入会按错误布局解释 scale。与标准 swizzled layout 的输出对比出现大幅差异和 NaN。

**处理决定：** 直接否决。这个方向必须同时修改 kernel 的 scale 解释，不能只切换 quantize 输出布局。

## 候选三：8×4 scale-factor layout

FlashInfer quantize 可以生成 8×4 scale layout，但当前使用的 CUTLASS backend 明确拒绝该布局，报错为 `Only TRTLLM FP4 GEMM supports 8x4 scale factor layout`。当前 DSpark 机器是 SM120，不能把 TRTLLM backend 直接当作可用替代路径。

**处理决定：** 不在当前配置上继续，除非后续实现 SM120 可用的 8×4 FP4 kernel。

## 候选四：cuDNN FP4 backend

在 `K=5120,N=17408` 的同一 microbenchmark 中，cuDNN FP4 GEMM 约为 0.23～0.27 ms，CUTLASS 约为 0.15～0.18 ms；它没有成为当前机器上的候选加速路径。

**处理决定：** 否决当前 DSpark 配置下的 cuDNN 替换。

## 为什么没有继续完整测试

完整矩阵需要多个服务进程、每格多个请求批次，运行成本很高。按照预先设定的筛选条件，候选必须先满足：

1. 输出和 acceptance 元数据正确；
2. microbenchmark 在目标 M 范围内方向稳定；
3. 服务 AB/BA 至少方向一致，不能依赖单一执行顺序。

小 M PDL 候选只满足前两项的一部分，服务 AB/BA 两轮方向不一致，因此继续跑完整矩阵不会增加可靠结论，只会增加一次性噪声。

完整 NVFP4 基线矩阵仍以 [NVFP4/FP8 同协议报告](./NVFP4_DSPARK_FP8_SAME_PROTOCOL_20261108.md) 为准。下一步真正值得做的是写一个针对实际 verify M 分布的 fused FP4 quantize + block-scale GEMM kernel，并用 Nsight Compute 分离 quantize、scale load、GEMM、epilogue 和 launch 时间，再决定是否投入完整服务测试。
