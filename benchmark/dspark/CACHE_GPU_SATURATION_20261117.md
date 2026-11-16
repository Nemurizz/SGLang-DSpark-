# DSpark 成对前缀缓存：GPU 饱和度对齐实验（2026-11-17）

## 这项实验补什么

第五项到达率回放已经模拟了夜间、日间、峰值和冷却流量，但当时的 GPU sampler 只留下了 startup 记录，不能回答“paired cache 的收益是否来自 GPU 变得更忙”。第六项复用同一条请求轨迹和相同的服务配置，在请求运行期间用独立 NVML 线程每 100 ms 采样 GPU busy、显存和功耗，并用单调时钟把采样点切到每个阶段。

full-prefill 和 paired cache 各运行一次，阶段顺序和第五项相同：`night_low` 1 req/s、`day_medium` 4 req/s、`peak_burst` 20 req/s、`cooldown` 2 req/s。每阶段请求数为 8/16/32/8，`p512/c4`，输出 128 token，Qwen3-14B + DSpark block7，BF16 target/KV，FA4 fixed64，CUDA Graph/compact scan/adaptive draft 关闭。

## 阶段级性能与 GPU 采样

| 阶段 | Full E2E 均值（ms） | Paired E2E 均值（ms） | E2E 变化 | Full p95（ms） | Paired p95（ms） | Full GPU busy 均值 / p95 | Paired GPU busy 均值 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| night_low | 4327.36 | 6686.07 | -54.51% | 6272.68 | 8403.71 | 54.82% / 99% | 43.22% / 94% |
| day_medium | 4886.12 | 5286.15 | -8.19% | 7454.25 | 8176.01 | 97.02% / 99% | 87.66% / 93% |
| peak_burst | 10231.91 | 10888.64 | -6.42% | 18285.12 | 18954.38 | 94.46% / 99% | 89.73% / 98% |
| cooldown | 2926.96 | 2608.34 | **+10.89%** | 3660.70 | 3270.80 | 88.62% / 95% | 93.00% / 99% |

总共 64 个请求按请求数加权后，full 的 E2E 均值为 7244.27 ms，paired 为 7927.66 ms，paired 整体慢 **9.43%**。paired 服务本轮记录到 22119 个命中 token、18 次 metadata eviction、0 次 partial recovery。

## GPU 饱和度能说明什么

`day_medium` 和 `peak_burst` 的 full GPU busy 均值已经接近 95%～97%，可以把它们视为本轮真正的高 GPU 忙碌阶段。可是这两个阶段 paired 的 GPU busy 反而低于 full，同时 E2E 还分别慢 8.19% 和 6.42%。这说明当前 paired 路径没有把高 GPU busy 自动转化成收益；cache lookup、readiness/metadata 更新、请求调度或 prefill 工作的改变可能让 GPU 出现空转，端到端时间由 CPU/调度链路限制。

`cooldown` 是相反的例子：paired GPU busy 均值 93.00%，高于 full 的 88.62%，E2E 快 10.89%。这里可以观察到缓存命中和 GPU 工作重叠，但单次复跑不能证明“GPU busy 超过某个阈值就应该开启 cache”。

## 与第五项同轨迹复跑的对照

第五项和本项使用同一条确定性 hot/cold 请求序列及四段到达率。两次结果方向并不稳定：

| 阶段 | 第五项 E2E 变化 | 第六项 E2E 变化 |
|---|---:|---:|
| night_low | -28.68% | -54.51% |
| day_medium | +5.36% | -8.19% |
| peak_burst | +5.19% | -6.42% |
| cooldown | -2.64% | +10.89% |

这不是实验失败，而是当前证据最重要的限制：一次 full/paired 对照不足以给出稳定的 cache admission 阈值。即使请求轨迹、并发和配置固定，GPU 时钟、JIT/allocator 状态、进程启动状态和系统噪声也会改变阶段级 E2E。第五项和第六项合在一起支持“需要重复轮次和配对统计”，不支持把某一次的 5% 或 10% 直接写成固定生产收益。

## DSpark 当前是否已经感知 GPU 饱和度

没有证据表明当前 DSpark planner 会读取本轮 NVML GPU busy，并据此自动打开或关闭 paired cache。planner 主要根据请求数、verify geometry、SPS additive step-time 向量和预算选择验证长度；paired cache 依据 prefix 命中、draft readiness、metadata 预算和 recovery 状态决定复用。GPU busy 采样是本实验外部加入的观测，不是现有控制回路的一部分。

如果要实现 GPU-aware admission，合理的外层控制应同时看：最近窗口的 GPU busy、server queue wait、到达率、实际 cache hit token、eviction/recovery 成本和 E2E p95，并设置滞回和最短保持时间。只看 GPU busy 会把“GPU 很忙但 CPU/缓存管理更慢”和“GPU 忙且 cache 真正节省 prefill”混在一起。

## 测量边界

- NVML 线程每 100 ms 采样一次，按服务阶段的单调时间窗口计算均值和 p95；显存和功耗原始 CSV 已保存。
- 本轮没有独立的 server queue-wait 字段，E2E 包含排队和执行；客户端只记录了计划发送时间与实际发送偏差。
- 每种模式只有一个服务对、每阶段 8～32 个请求，没有 bootstrap 置信区间，GPU busy p95 也不是请求级 p95。
- NVML 数据能证明“采样窗口内 GPU 忙碌程度”，不能单独证明 cache lookup 或 planner 是因果瓶颈；还需要 CPU wall、scheduler queue 和 GPU kernel 时间线。

## 结论

第六项补齐了第五项缺失的 GPU 观测：高到达率阶段确实对应接近饱和的 GPU busy，但 paired cache 的 E2E 方向仍可能相反，说明 DSpark 当前没有基于 GPU busy 的自动 cache 策略，且 cache 管理成本可能把 GPU 节省抵消。第五、六项的同轨迹结果不稳定，下一步应做至少 5～10 轮随机化 ABBA，配对记录 GPU busy、CPU planner、queue wait、hit/eviction/recovery、TTFT、E2E 和 p95，再决定是否实现 GPU-aware admission。

## 原始证据

- 远端结果目录：`/root/autodl-fs/dspark-gpu-saturation-20261117`
- 本地结果：[results/cache-gpu-saturation-20261117/result.json](results/cache-gpu-saturation-20261117/result.json)
- NVML 原始采样：[results/cache-gpu-saturation-20261117/nvml/full.csv](results/cache-gpu-saturation-20261117/nvml/full.csv)、[paired.csv](results/cache-gpu-saturation-20261117/nvml/paired.csv)
- 阶段时间和请求摘要：[results/cache-gpu-saturation-20261117/paired-summary.json](results/cache-gpu-saturation-20261117/paired-summary.json)

## 任务状态

第六个任务已完成。后续任务尚未开始。
