# DSpark 成对前缀缓存：生产形状到达率回放（2026-11-17）

## 这项实验要回答什么

前面的实验把请求按固定并发批次发出，能够说明缓存命中、淘汰和 partial recovery 是否发生，却不能回答一个线上更常见的问题：请求不是同时到达的，白天、夜间和突发流量下，缓存收益是否会改变。

本实验固定同一条请求轨迹，只改变到达节奏。每个阶段仍然使用 `p512/c4`、输出 128 token、Qwen3-14B + DSpark block7，full-prefill 和 paired cache 各启动一个独立服务。请求由同一个确定性 hot/cold 序列构造，客户端按固定时间戳发出；因此两种服务看到的请求顺序、输入和到达间隔一致。

| 阶段 | 到达率 | 请求数 | 间隔 | hot 请求比例 | 代表场景 |
|---|---:|---:|---:|---:|---|
| `night_low` | 1 req/s | 8 | 1.00 s | 75% | 夜间低负载 |
| `day_medium` | 4 req/s | 16 | 0.25 s | 50% | 日间稳定流量 |
| `peak_burst` | 20 req/s | 32 | 0.05 s | 75% | 短时突发峰值 |
| `cooldown` | 2 req/s | 8 | 0.50 s | 87.5% | 峰值后的回落 |

服务配置使用当前已通过验证的 FA4 fixed64 profile（`0577109`）、BF16 target/KV、CUDA Graph/compact scan/adaptive draft 关闭，metadata budget 为 25%。`full` 关闭 radix cache，`paired` 开启 DSpark target/draft 成对缓存。每个阶段从空缓存开始运行一次，阶段之间不清空 paired cache。

## 结果

`E2E change = 1 - paired/full`，正数表示 paired 更快。p95 是请求级 p95。命中和淘汰为 paired 服务该阶段前后的计数差。

| 阶段 | Full E2E 均值（ms） | Paired E2E 均值（ms） | E2E 变化 | Full p95（ms） | Paired p95（ms） | 命中 token | metadata eviction | partial recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| night_low | 4405.81 | 5669.31 | -28.68% | 6352.76 | 7340.40 | 2038 | 0 | 0 |
| day_medium | 4886.21 | 4624.24 | **+5.36%** | 7463.93 | 7084.19 | 4151 | 0 | 0 |
| peak_burst | 10369.77 | 9831.11 | **+5.19%** | 18279.79 | 17437.52 | 12343 | 16 | 0 |
| cooldown | 2574.52 | 2642.44 | -2.64% | 3222.53 | 3327.33 | 3587 | 2 | 0 |

64 个请求按请求数加权后，full 的 E2E 均值为 7278.98 ms，paired 为 7110.58 ms，paired 整体快 **2.31%**。这不是把四个阶段的 p95 简单平均，而是只对均值按请求数加权；每个阶段的 p95 应单独看。

## 如何理解结果

低负载阶段 paired 慢 28.68%。此时请求之间有足够间隔，GPU 没有持续排队，缓存查找、readiness 维护和成对 metadata 生命周期成本反而更容易显现。缓存存在不代表低负载服务一定值得开启。

日间中负载和 20 req/s 峰值阶段分别快 5.36% 和 5.19%，同时 p95 分别下降 5.09% 和 4.61%。这说明当请求开始重叠、prefill 工作和排队成为主要成本时，稳定的前缀复用能够抵消部分 paired bookkeeping。峰值阶段还产生了 16 次 metadata eviction，但没有 partial recovery，说明这条轨迹下淘汰尚未破坏可直接复用的 draft readiness。

冷却阶段 paired 慢 2.64%。峰值留下的缓存状态仍在，但流量下降后节省的 prefill 工作减少，固定的缓存维护成本再次占据更大比例。它和夜间阶段共同说明：缓存策略应该感知负载，而不是只根据静态命中率常开。

## 对 DSpark 负载感知的结论

本实验没有发现现有 DSpark planner 会根据 GPU 利用率或“白天/夜间”自动切换 paired cache。DSpark 的动态 planner 主要依据请求数、verify geometry、SPS step-time 代价向量和预算做验证长度决策；paired prefix cache 则依据 prefix 命中、draft readiness、metadata 预算和恢复状态做复用决策。两者之间没有一个已验证的“GPU 饱和度高就开启 cache、低就关闭 cache”的控制环。

因此，当前结果支持一个工程上的外层策略：由服务层根据近期到达率、排队长度、GPU busy 和缓存收益窗口做 admission 或模式选择；DSpark 内部继续负责 verify budget 和安全的 target/draft 复用。切换策略必须先经过滞回和冷却时间，避免请求率在阈值附近抖动导致反复开关。

## GPU 饱和度的测量边界

本轮客户端记录了固定到达时间、客户端发送偏差和 E2E/TTFT/TPOT，但远端 `GPUSampler` 输出只标记为 `startup`，没有形成按阶段划分的 GPU busy 时间序列。因此“peak_burst 是高负载阶段”是由 20 req/s 到达轨迹和明显升高的 TTFT/E2E p95 证明的，不能写成已经测得 GPU 利用率达到某个百分比。下一轮若要做 GPU-aware admission，应在服务进程外用独立的 NVML 或 `nvidia-smi dmon` 采样器，保存每个阶段的 GPU busy、显存、功耗和时序，并与服务请求时间线对齐。

## 结论边界与下一步

这次回放说明 paired cache 的收益随到达率变化：低负载下可能明显拖慢，中高负载下在本配置和合成轨迹中约有 5% 的阶段级收益，整体 64 请求加权收益为 2.31%。这不是生产阈值，也不是稳定提速保证；实验只有一个服务对、每阶段 8～32 个请求，输入长度固定为 512，轨迹是确定性 hot/cold 构造，未采集独立的 GPU busy 时间序列。

后续真正做 GPU-aware cache admission 时，应在相同请求 trace 下增加 5～10 个随机化重复轮次，至少覆盖不同 metadata budget、输入长度和峰值持续时间，并把 NVML GPU busy、server queue wait、cache hit、eviction、recovery 和 E2E 逐请求配对。只有当收益在不同重复轮次和到达形状下都稳定，才值得把外层切换策略接入服务。

## 原始证据

- 远端结果目录：`/root/autodl-fs/dspark-cache-arrival-replay-20261117`
- 本地原始结果：[results/cache-arrival-replay-20261117/result.json](results/cache-arrival-replay-20261117/result.json)
- 固定到达轨迹：[results/cache-arrival-replay-20261117/protocol.json](results/cache-arrival-replay-20261117/protocol.json)
- 阶段摘要：[results/cache-arrival-replay-20261117/paired-summary.json](results/cache-arrival-replay-20261117/paired-summary.json)

## 任务状态

第五个任务已完成。下一项任务尚未开始。
