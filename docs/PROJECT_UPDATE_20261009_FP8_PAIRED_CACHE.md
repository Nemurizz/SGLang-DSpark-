# DSpark 项目阶段性更新：FP8、paired radix cache 与稳定性边界

日期：2026-10-09

当前提交分支：`exp/quant-w8a8-fp8-native-target-bf16-draft`

`main` 没有修改。本次更新只在量化实验分支上增加可回退的 benchmark 和实验记录。

## 1. 为什么继续做这组实验

前面的 DSpark 工作已经解决了几个比 kernel microbenchmark 更重要的问题：

- acceptance 和 prefix-cache 的确定性保护；
- target KV 与 draft KV 的 paired readiness、fingerprint 和 partial recovery 语义；
- compact row-index reuse、可选 CUDA compact/scan 和动态 draft length 的安全边界；
- ragged verify graph 的 capability 检查和 exact capture/runtime geometry guard。

这些改动保证了“命中 cache 时不会错误复用 draft KV”，但还没有回答一个实际部署问题：

> 真实流量的前缀复用率到底要多高，paired cache 才能把省掉的 prefill 变成端到端收益？

同时，FP8 是目前比手写 GEMM、CUTLASS epilogue probe、采样融合和 ragged graph 更容易在
稳定态服务中验证的方向，所以先在独立分支上做 FP8 与 paired cache 的组合测量，没有把
未经证明的 kernel 实验混进主线结论。

## 2. 本阶段改动

### `2d95aac` / `f8c21c0` / `b097d3b`

- 增加高并发 BF16/FP8 counter-balanced A/B harness；
- 让 paired-cache 模式真正移除 `--disable-radix-cache`，并设置
  `SGLANG_DSPARK_DISABLE_RADIX_CACHE=0`、`SGLANG_DSPARK_PREFIX_CACHE=1`；
- 保留 FP8 sidecar、target/draft profile 和 runtime attestation，避免“命令行写了 FP8
  但实际服务仍是 BF16”的假实验。

### `57e0be1` / `fa3b6b1`

- 增加 standard CUDA Graph smoke 选项，但默认仍然关闭；
- 记录 4/8/16 并发、512/1024/2048/8192 prompt 的 FP8 结果，以及 paired cache 矩阵。

### `d5127a5`

- 增加 `benchmark/dspark/cache_hit_rate_sweep.py`；
- 增加 `benchmark/dspark/CACHE_HIT_RATE_SWEEP_RESULT_20261009.md`；
- 固定 BF16、FA4、graph off、p512/c4、输出 128，在完整 prefill 与 paired cache 之间
  扫描 0/25/50/100% 的显式完整前缀复用率。

## 3. 测试协议

- 模型：Qwen3-14B target，DSpark draft；
- dtype：BF16（本组用于隔离 cache，不把 FP8 算术差异混进来）；
- attention：FA4；
- CUDA Graph：关闭；ragged graph、compact CUDA scan、dynamic draft length、KV quantization
  均关闭；
- prompt：512 tokens；并发：4；`max-running-requests=4`；
- 每档 2 个 warmup batch、16 个 measured requests、输出 128 tokens；
- full 模式和 paired 模式使用同一请求序列；
- 记录 TTFT、TPOT、E2E、p95、GPU 显存、acceptance、target/paired hit 和 recovery。

## 4. 结果

| 显式完整前缀复用率 | target/paired token hit | paired hit length | full E2E mean | paired E2E mean | E2E reduction | full/paired E2E p95 | partial recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1.93% / 1.93% | 9.9 | 6133.55 ms | 6059.79 ms | 1.20% | 9812.15 / 9733.59 ms | 0 |
| 25% | 26.46% / 26.46% | 135.2 | 6189.69 ms | 6192.87 ms | -0.05% | 10020.61 / 10031.65 ms | 0 |
| 50% | 50.97% / 50.97% | 260.4 | 6070.68 ms | 5967.78 ms | 1.69% | 9621.49 / 9483.90 ms | 0 |
| 100% | 100% / 100% | 511.0 | 6134.35 ms | 5989.07 ms | 2.37% | 9830.70 / 9642.69 ms | 0 |

“0%”表示没有重复完整 prompt。由于测试 prompt 共享统一模板，仍然存在约 10 个 token
的结构性公共前缀，所以 token-level hit 不是绝对 0%。这反而说明线上统计必须同时记录
完整 prefix hit 和 token hit，不能只看请求数命中率。

100% 命中时服务日志明确显示：

```text
paired: new-token=1, cached-token=511
full:   new-token=512, cached-token=0
```

但是 E2E 只减少 2.37%。在该协议中，decode、speculative verify 和排队时间已经占据大部
分延迟，prefill 节省被 Amdahl 定律限制。25% 档略慢 0.05%，说明低命中率时 radix 查找、
metadata 维护和 cache 管理开销可能抵消收益。

两种模式的 GPU 峰值显存都约 38955 MiB。paired cache 的主要价值是减少重复 prefill，
不是自动降低静态权重或 KV 分配。

## 5. FP8 与 cache 的组合结论

此前的高并发 compute-isolation 测试（cache off、graph off）观察到 FP8 相对 BF16 的
E2E 收益约为 10–32%，8K prompt/c4-c8 的收益约 31–32%。

paired cache 打开后，FP8 仍然通常是最快的绝对配置，但 FP8 相对 BF16 的增量收益会被
cache 掩盖：paired-cache 矩阵中只有部分 cell 同时满足“E2E 平均收益 >1% 且 TTFT/TPOT/E2E
p95 无回退”。因此不能写成“cache 让量化收益更高”，准确说法是：

> FP8 + paired cache 是当前测到的最佳组合；cache 命中越高，绝对延迟越低，但留给 FP8
> 继续节省的 prefill 部分越少。

## 6. 遇到的问题与处理

### 6.1 以为打开 sidecar 就是打开 radix cache

第一次 paired-cache harness 仍继承了稳定 profile 的 `--disable-radix-cache`，结果
服务实例化的是 ChunkCache，没有 `dspark_prefix_cache` sidecar 统计。

处理方式：显式删除该命令行参数，并同时设置 radix/cache 环境变量；之后通过
`/server_info` 强制检查 `disable_radix_cache=false`，否则立即 safety stop。

### 6.2 CUDA Graph smoke 被固定 FA4 guard 拒绝

标准 graph smoke 确实开始捕获 58 个 target prefill shape，但随后被固定 FA4 deterministic
profile 拒绝：该 profile 要求 graph 全关。之前放宽 geometry guard 曾导致串行/并发结果
不一致，因此没有移除 guard，也没有把这次 smoke 误报成 OOM 或性能结论。

处理方式：graph 继续保持 opt-in；当前稳定 A/B 仍使用 graph off。后续若重新研究 graph，
必须建立 graph catalog 或 page-table 重排，并逐 geometry 做 token-by-token correctness。

### 6.3 新 harness 第一次启动失败

第一次 sweep 覆盖了模板中的 `PYTHONPATH`，丢失共享 SGLang 源码路径，服务在启动阶段报
`ModuleNotFoundError: sglang.srt.arg_groups`。没有发出任何推理请求。

处理方式：恢复 `/root/autodl-tmp/src/sglang-main/python`，再追加本实验分支的 benchmark
路径后重跑。

### 6.4 miss prompt 池过短

第一次修复后的 harness 在四档复用率累计运行时耗尽 unique miss prompt 池，触发
`IndexError`。这是测试工具问题，不是服务错误。

处理方式：扩大确定性 miss 池，删除失败的临时输出后从头重跑；最终四档均完成，服务日志
无 CUDA error、illegal memory access 或 scheduler exception。

## 7. 正确性检查

- paired 模式四档 `match_calls=16`；
- `paired_over_target_token_coverage=1.0`；
- partial/full recovery 全部为 0（本实验没有主动制造 draft readiness 缺失）；
- full 与 paired 对相同 prompt 的 output token 和 acceptance 字段逐请求比较，0/25/50/100%
  四档均为 0 mismatch；
- acceptance 在不同复用率之间会随 prompt mix 变化，不能把不同 prompt 集的 acceptance
  差异误判为 cache 错误。

## 8. 当前不能证明的内容

- 不能把统一模板的 100% 命中结果外推到真实用户流量；
- 没有证明 partial recovery 在真实 eviction 压力下能产生收益；
- 没有证明 ragged graph 在 exact geometry 之外安全 replay；
- 没有证明手写 GEMM、CUTLASS epilogue、采样融合可以带来稳定态端到端收益；
- 没有把单 kernel microbenchmark 当作服务级收益。

## 9. 最终结论与后续方向

1. 当前最可靠的绝对性能组合是 FP8 target + BF16 draft + paired cache，但 cache 的收益高度
   依赖真实 prefix reuse 分布。
2. 当显式完整前缀复用低于约 25% 时，paired cache 可能没有收益；约 50% 以上才开始在本
   协议中稳定超过 1%。这不是理论阈值，只是当前 p512/c4 配置下的观测边界。
3. 继续改 cache 数据结构前，应该先从真实请求 trace 得到 prefix reuse CDF，再做 c8/c16、
   不同 output length 和不同 queue pressure 的确认矩阵。
4. CUDA Graph、手写 GEMM、CUTLASS epilogue 和采样融合暂时保持独立实验，不合并到本结论。
5. 如果真实 trace 的命中率长期低于 25%，下一步优先优化 FP8/metadata/调度，而不是继续
   扩大 paired-cache 代码复杂度。

