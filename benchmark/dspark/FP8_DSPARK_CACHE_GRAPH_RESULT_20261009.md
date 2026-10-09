# DSpark paired-cache / CUDA graph 高并发结论（2026-10-09）

这份报告补充 native W8A8 FP8 高并发 compute-isolation 结果，回答 cache 和 graph 真正打开后是否仍成立。

## Paired cache 主矩阵

服务实际配置为 `disable_radix_cache=false`、`SGLANG_DSPARK_PREFIX_CACHE=1`，不是仅设置 sidecar 环境变量。graphs、ragged DSpark graph、compact scan、adaptive draft 和 KV quantization 仍关闭。主矩阵为 `p512/p1024/p2048 × c4/c8/c16` 的合法容量 cell，32 个稳态请求，BF16→FP8→FP8→BF16。

所有 8 个 cell 的 cache 统计都一致：

- target token hit rate = 100%
- paired token hit rate = 100%
- paired hit length = prompt length − 1
- partial recovery calls/tokens = 0
- full recovery calls/tokens = 0
- 每个请求只发生一次 match invocation

但是 cache 命中后，FP8 相对 BF16 的边际收益比 full-prefill 小很多。严格要求“AB 和 BA 两个顺序都 E2E mean >1%，且 TTFT/TPOT/E2E p95 都不退化”时，稳定通过的只有：

| Cell | 结果 | 原因 |
|---|---|---|
| p512_c4 | 通过 | 两个顺序均通过，E2E mean +6.10–8.17% |
| p1024_c4 | 通过 | 两个顺序均通过，E2E mean +5.25–7.34% |
| p1024_c16 | 通过 | 两个顺序均通过，E2E mean +3.68–13.54% |
| p512_c16 | 不宣称 | BA 的 TTFT p95 ratio=1.012 |
| p512_c8 | 不宣称 | 两个顺序 TPOT p95 ratio=1.029/1.055 |
| p1024_c8 | 不宣称 | 两个顺序 TPOT p95 ratio=1.202/1.234 |
| p2048_c4 | 不宣称 | BA E2E mean gain 仅 0.66% |
| p2048_c8 | 不宣称 | TPOT p95 ratio=1.048/1.087，BA gain 仅 0.29% |

这不是 cache correctness 失败；是 cache 把 prefill 优势消除后，FP8 decode/queueing 的边际收益不足以覆盖严格 tail gate。

## Cache on/off 的直接对照

在 `p512_c4`、相同 32 请求/4 warmup 配置下，paired cache 相对 full-prefill 的独立 ABBA 对照为：

| 精度 | E2E mean 降幅 | E2E p95 降幅 | paired hit |
|---|---:|---:|---:|
| BF16 | 20.9–24.8% | 17.6–21.5% | 100% |
| FP8 | 15.0–16.8% | 12.4–14.1% | 100% |

因此 paired cache 本身有明确端到端价值；但不能把它和 FP8 compute gain 直接相加。两者组合后的最终收益要按 cell 单独验证。

## CUDA graph smoke

尝试过 `--enable-standard-graphs` + paired cache + `p512_c4` smoke。服务捕获了 58 个 target prefill graph shape，但在启动阶段被 `fixed_fa4_policy_hook` 拒绝：该稳定 FA4 profile 明确要求 deterministic inference 下所有 graph 关闭。请求尚未执行，因此：

- 没有 graph correctness 通过证据；
- 没有 graph latency/显存收益结论；
- 不能删除 guard 或绕过 hook 重试。

要继续 graph，必须先建立独立的 graph-compatible FA4 policy/geometry catalog，并对每个 geometry 做 token-level correctness 和 ABBA；当前版本应继续保持 graph off。

## 最终决策

- **值得保留**：paired cache；它在所有已测主矩阵 cell 都命中正确，且 cache on/off 有 15–25% 级别的 E2E 收益。
- **条件保留**：native FP8；full-prefill 高并发收益很强，但在 paired-cache 命中后只在 `p512_c4`、`p1024_c4`、`p1024_c16` 通过严格组合门槛。
- **暂不打开**：CUDA graph；当前稳定 FA4 guard 会在启动阶段拒绝，且历史上放宽 guard 曾导致并发输出不一致。
- **未证明**：8K paired-cache、partial recovery 的真实高并发收益，以及 cache+graph 组合收益。

证据目录：

- paired cache 主矩阵：`/root/autodl-tmp/dspark-quant-fp8-paired-high-conc-20261009-h74`
- cache focus ABBA：`/root/autodl-tmp/dspark-quant-fp8-paired-high-conc-20261009-h72`
- full-prefill focus 对照：`/root/autodl-tmp/dspark-quant-fp8-full-high-conc-20261009-h73`
- graph smoke：`/root/autodl-tmp/dspark-quant-fp8-graph-smoke-20261009-h75`
