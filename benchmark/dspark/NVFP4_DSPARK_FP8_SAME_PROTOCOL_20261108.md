# NVFP4 与 FP8 在 DSpark 下的同协议对照（2026-11-08）

## 结论先说

在同一台 RTX PRO 6000 Blackwell、同一份 Qwen3-14B target、同一份 DSpark block7 draft、同一套 FA4 fixed64 验证路径下，FP8 是当前这套 DSpark 服务的性能方案，NVFP4 是显存容量方案：

- FP8 在 8 个矩阵格子的平均 E2E gain 为 **+5.12%**，每个格子均为正收益；但 p1024/c16 的收益最大，长上下文或高并发时收益收窄。
- NVFP4 在同样 8 个格子中平均 E2E gain 为 **−69.85%**，也就是端到端时间约为 BF16 的 1.5～1.9 倍；TTFT p95 平均为 BF16 的 **1.652 倍**，TPOT p95 平均为 **1.685 倍**。
- NVFP4 的平均设备峰值显存为 **23132 MiB**，FP8 为 **28952 MiB**，相对 BF16 的约 39～42 GiB 峰值分别少约 **29%** 和 **43%**。NVFP4 的主要价值在于把模型放进更小的显存预算，而不是在当前 DSpark 验证几何下提速。

这组数据不支持“NVFP4 在 DSpark 下能加速”的表述；更准确的工程结论是：**当前版本优先用 FP8 保性能，只有显存容量成为硬约束时才考虑 NVFP4，并且需要为 NVFP4 单独优化小 M、非规则 verify GEMM 路径。**

## 实验怎么保证可比

两组实验使用相同的 8 个格子：`p512/p1024/p2048 × c4/c8/c16`（去掉 p2048/c16 这一超容量组合），每个格子 2 个 warmup batch、32 个 measured requests，输出长度 128。请求数据、请求顺序、到达轨迹、前缀缓存模式和容量固定；每组都执行 AB 与 BA 两个顺序，避免把启动顺序或热缓存偏差误判为量化收益。

共同配置为：paired prefix cache、`max_total_tokens=32768`、BF16 draft、DSpark block7、FA4 fixed64、CUDA Graph/compact scan/adaptive draft/KV quantization 关闭。FP8 与 NVFP4 都只替换 target 的权重执行路径，draft、SPS 表和服务调度保持不变。跨 recipe 不要求 token 逐字相等，但每个 recipe 内部都检查 token 与 DSpark acceptance 元数据的一致性。

## 逐格结果

表中的 E2E gain 定义为 `1 - quantized_e2e / BF16_e2e`；正数表示更快，负数表示更慢。p95 ratio 小于 1 表示量化版本更快。

| 输入/并发 | FP8 E2E | NVFP4 E2E | FP8 TTFT p95 | NVFP4 TTFT p95 | FP8 TPOT p95 | NVFP4 TPOT p95 | BF16/FP8/NVFP4 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| p512/c4  | +7.13% | −70.48% | 0.923 | 1.599 | 0.942 | 1.632 | 39261 / 27193 / 21447 |
| p512/c8  | +3.67% | −61.06% | 0.934 | 1.375 | 1.042 | 1.746 | 39725 / 27657 / 21796 |
| p512/c16 | +4.97% | −60.45% | 0.995 | 1.610 | 0.942 | 1.600 | 41257 / 29189 / 23226 |
| p1024/c4 | +6.30% | −79.96% | 0.834 | 1.801 | 0.838 | 1.857 | 41257 / 29189 / 23227 |
| p1024/c8 | +5.37% | −62.05% | 0.972 | 1.609 | 1.218 | 1.677 | 41257 / 29189 / 23227 |
| p1024/c16| +8.61% | −50.71% | 0.863 | 1.530 | 0.924 | 1.271 | 41801 / 29733 / 24043 |
| p2048/c4 | +2.76% | −88.98% | 0.891 | 1.913 | 0.905 | 1.975 | 41801 / 29733 / 24043 |
| p2048/c8 | +2.13% | −85.15% | 0.860 | 1.776 | 1.067 | 1.727 | 41801 / 29733 / 24043 |

FP8 的 p95 不是每一项都优于 BF16：例如 p1024/c8 的 TPOT p95 ratio 为 1.218。因此 FP8 的结论是“平均 E2E 有小幅稳定收益”，不是所有尾延迟指标都单调改善。NVFP4 则在每个格子同时出现 E2E、TTFT 或 TPOT 的明显回退。

## 为什么 NVFP4 会慢

这不是“4 bit 一定更快”的简单问题。当前 DSpark 的 verify 阶段每一步是小而不规则的 M，target 还要配合 speculative acceptance、缓存读写和调度。ModelOpt NVFP4 路径需要 FP4 block scale、量化布局和对应的反量化/缩放处理；在小 M 下，scale 读取、布局转换和 kernel dispatch 的固定开销占比上升，权重带宽节省无法覆盖这些开销。FP8 走的是 SGLang 原生 W8A8 scaled GEMM，和现有 DSpark 的 verify 几何更匹配。

Acceptance 也不是主要解释：8 个格子的平均 acceptance 为 BF16 0.356、FP8 0.354、NVFP4 0.327。NVFP4 确实略低，但不足以解释 1.5～1.9 倍的端到端回退；主要差异来自 target 量化 kernel 路径与 DSpark 小批量 verify 的不匹配。

## 后续优化方向

如果目标是继续使用 NVFP4，应把优化目标从“压低权重位宽”改成“让 NVFP4 的执行形状适配 DSpark”：针对固定的 verify M 做专用 FP4 block-scale GEMM，缓存或融合 scale/layout 转换，减少每 step 的 dispatch，并对 p2048/c4、p2048/c8 这类长上下文格子单独做 kernel 与 launch 调优。只有在这些开销被量化并消除后，才值得重新评估 NVFP4 的服务吞吐。

## 证据文件

- `FP8_NVFP4_COMPARISON.md`：逐格汇总表。
- `fp8_nvfp4_comparison.json`：FP8 与 NVFP4 的配对结构化数据。
- `comparisons.json`、`cells.jsonl`：NVFP4 原始 AB/BA 结果。
- `protocol.json`、`status.json`：实验协议与完成状态。
- `SHA256SUMS`：归档文件校验值。

原始运行目录位于远端 `/root/autodl-tmp/dspark-quant-nvfp4-paired-high-conc-20261108-h77`；本目录保存了用于复核的结果摘要和结构化数据。
