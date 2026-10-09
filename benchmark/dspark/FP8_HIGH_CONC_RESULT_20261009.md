# 原生 W8A8 FP8 高并发结论（2026-10-09）

## 结论

在同一 serving 配置、同一模型、同一 DSpark、graphs/radix cache/compact scan/adaptive draft/KV quantization 全关闭的条件下，原生 W8A8 FP8 target + BF16 draft 在高并发下通过了独立精度内 correctness gate，并在所有已测合法容量 cell 上达到“E2E 平均收益 >1%、TTFT/TPOT/E2E p95 不退化”。

主矩阵覆盖 `p512/p1024/p2048 × c4/c8/c16`，但 `p2048_c16` 不属于 `max_total_tokens=32768` 的合法容量 cell，因此没有被伪造为失败样本。另跑了独立的 `p8192_c4/c8` 长上下文 suite，容量设为 131072。

## 协议

- 顺序：BF16_A → FP8_A → FP8_B → BF16_B。
- 主矩阵：`max_total_tokens=32768`，每 cell 32 个稳态请求、2 个并发 warmup batch、输出 128 token。
- 长上下文：`max_total_tokens=131072`，覆盖 `p8192_c4/c8`。
- `p512_c4` 另做了稳定性复测：8 个 warmup batch、64 个稳态请求；用该结果替换首轮短 warmup 的漂移样本。
- 每种精度只与自己的冷参考输出比较，要求 output token IDs 和五个 DSpark acceptance 字段完全一致；不要求 BF16 与 FP8 跨精度逐 token 相同。
- 每个 arm 的 cache/retraction 检查均为零；本 suite 关闭 radix/paired cache，因此没有 shadow hit 或 paired hit 结论。

## E2E/p95 结果

收益为 FP8 相对 BF16 的 E2E mean；范围是 AB/BA 两个顺序。p95 ratio 小于 1 表示 FP8 更快。

| Cell | E2E mean gain | E2E p95 gain | TTFT p95 ratio | TPOT p95 ratio | BF16/FP8 peak MiB | BF16/FP8 acceptance |
|---|---:|---:|---:|---:|---:|---:|
| p512_c4（稳定复测） | 14.18–16.74% | 14.88–17.47% | 0.811/0.787 | 0.882/0.858 | 40863/28795 | 33.30%/34.57% |
| p512_c8 | 10.38–14.29% | 9.22–13.33% | 0.857/0.821 | 0.997/0.948 | 41247/29179 | 35.81%/34.76% |
| p512_c16 | 11.85–16.55% | 14.86–19.36% | 0.885/0.839 | 0.938/0.882 | 41247/29179 | 34.57%/33.65% |
| p1024_c4 | 20.42–22.62% | 19.41–21.89% | 0.731/0.717 | 0.793/0.771 | 41791/29723 | 37.12%/37.69% |
| p1024_c8 | 19.44–20.03% | 18.12–18.97% | 0.817/0.811 | 0.976/0.962 | 41791/29723 | 35.57%/34.20% |
| p1024_c16 | 23.68–24.46% | 25.70–26.60% | 0.724/0.715 | 0.822/0.806 | 41791/29723 | 31.61%/35.32% |
| p2048_c4 | 22.28–23.63% | 15.73–17.14% | 0.848/0.833 | 0.705/0.692 | 41791/29723 | 38.47%/36.86% |
| p2048_c8 | 24.19–24.63% | 25.86–26.30% | 0.683/0.680 | 0.790/0.786 | 41791/29723 | 38.40%/35.86% |
| p8192_c4（131K capacity） | 31.53–31.88% | 29.28–29.87% | 0.687/0.683 | 0.672/0.669 | 56507/46237 | 34.72%/33.27% |
| p8192_c8（131K capacity） | 32.12–32.37% | 29.33–29.68% | 0.682/0.680 | 0.612/0.610 | 56507/46239 | 34.30%/32.77% |

所有上表 cell 的 AB、BA 两个顺序都满足 E2E mean >1%，且 TTFT/TPOT/E2E 三项 p95 ratio 均不超过 1。

## 怎样理解“行”和“不行”

行：512–2K prompt 的 c4/c8/c16 合法容量场景，以及在 131K 容量下的 8K c4/c8。高并发下收益主要来自 target W8A8 FP8 计算和较低的 target 权重/运行显存；主矩阵峰值显存约下降 28.9–29.3%，8K suite 下降约 18.2%。吞吐提升大致为 1.14–1.43×，长上下文约 1.41–1.43×。

不应宣称的场景：`p2048_c16`、`p8192_c16` 没有在当前容量约束下测试；它们需要更大的 `max_total_tokens`，不能把容量不足误判为 FP8 性能失败。`p512_c8` 和 `p1024_c8` 的 TPOT p95 接近持平（最弱为 0.997、0.976），所以不能说所有高并发 cell 的 decode tail 都大幅改善；它们的 E2E 优势主要来自 TTFT 和整体 batch 计算。acceptance 也不是单调提高，部分 cell FP8 低约 0.9–2.5 个百分点，但没有造成 token/acceptance 稳定性错误，且 E2E 仍通过门槛。

这组数据不证明 paired-cache/shadow-cache 收益，因为计时协议明确关闭了 cache；也不证明打开 ragged graph、compact scan、adaptive draft 或 KV quantization 后仍保持同样收益。

## 证据位置

- 主矩阵：`/root/autodl-tmp/dspark-quant-fp8-high-conc-20261009-h67b`
- 8K suite：`/root/autodl-tmp/dspark-quant-fp8-high-conc-20261009-h68`
- p512_c4 稳定复测：`/root/autodl-tmp/dspark-quant-fp8-high-conc-20261009-h69`
- harness：`benchmark/dspark/quant_high_conc_ab.py`

这些结果是当前 native FP8 路线的高并发 serving 证据，不是完整质量榜单，也不是 paired-cache 的证据。
