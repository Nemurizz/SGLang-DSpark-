# Paired-cache prefix reuse-rate sweep

日期：2026-10-09。固定 BF16、FA4、graph off、p512/c4、输出 128、16 个稳态请求、2 个 warmup batch；同一请求序列分别运行 full-prefill（radix disabled）和 paired cache。

## 结果

| 显式完整前缀复用率 | paired target token hit | paired hit length | full E2E mean (ms) | paired E2E mean (ms) | E2E reduction | full/paired E2E p95 (ms) | partial recovery | acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1.93% | 9.9 | 6133.55 | 6059.79 | 1.20% | 9812.15 / 9733.59 | 0 | 0.3427 |
| 25% | 26.46% | 135.2 | 6189.69 | 6192.87 | -0.05% | 10020.61 / 10031.65 | 0 | 0.3250 |
| 50% | 50.97% | 260.4 | 6070.68 | 5967.78 | 1.69% | 9621.49 / 9483.90 | 0 | 0.3440 |
| 100% | 100.00% | 511.0 | 6134.35 | 5989.07 | 2.37% | 9830.70 / 9642.69 | 0 | 0.3330 |

“0%”表示没有重复完整 prompt，而不是所有 token 都完全不共享；统一的 prompt 模板仍留下约 10 个 token 的结构性前缀，因此 target token hit 为 1.93%。这正是生产请求模板可能产生的短公共前缀效应。

paired 模式的 `match_calls=16`、`paired_over_target_token_coverage=1.0`；四档均无 partial/full recovery。full/paired 共同 prompt 的 output token 和 acceptance 字段逐请求比较，0/25/50/100% 四档均为 0 mismatch。

## 解释

缓存确实生效：100% 档的 prefill 日志是 `new-token=1, cached-token=511`，full 模式仍是 `new-token=512`。但本协议中输出 128、并发 4、`max-running-requests=4`，总延迟主要由 decode、speculative verify 和排队构成；因此 prefill 省下来的部分只占总 E2E 的小比例。25% 档的 -0.05% 也说明 radix 查找/元数据维护成本可能抵消少量命中收益。

显存峰值两种模式均约 38955 MiB，paired cache 没有降低静态权重/KV 分配；它的价值是减少重复 prefill，而不是自动降低模型驻留显存。

该结果不能外推到所有并发和输出长度；它回答的是“请求相似度变化时 cache 是否命中、以及高并发下能否转化为 E2E 收益”。
