# DSpark 成对前缀缓存：预热、命中、淘汰与回访实验（2026-11-17）

## 这项实验要回答什么

前两项缓存实验分别回答了“功能能不能正确执行”和“发生 partial recovery 时 worker 是否真的执行了恢复”。第三项 mixed hit-rate sweep 只改变请求复用率，尚未覆盖缓存长期运行时更容易出问题的生命周期：缓存先被预热，随后被大量新前缀挤压，原来的热前缀再回来时还能不能继续命中，命中不足时能不能恢复 draft 状态。

因此本实验把一条固定请求轨迹拆成五个阶段：

1. `cold`：8 个从未出现过的冷前缀，观察冷启动成本；
2. `prime`：4 个热前缀第一次进入缓存，建立 target 与 draft 的 paired metadata；
3. `warm`：重复这 4 个热前缀两轮，观察稳定命中；
4. `evict`：连续加入 32 个新冷前缀，制造 metadata 容量压力；
5. `rehit`：再次访问热前缀，观察仍然命中的部分，以及触发的 partial recovery。

每个阶段的请求顺序、输入构造和并发都固定。`full` 服务关闭 radix cache，作为 full-prefill 对照；`paired` 服务开启 DSpark target/draft 成对前缀缓存。两套服务独立启动，配置均为当前通过验证的 FA4 fixed64 profile（`0577109`）、Qwen3-14B、DSpark block7、BF16 target/KV、CUDA Graph/compact scan/adaptive draft 关闭，`p512/c4`，输出 128 token。每阶段分别为 8/4/8/32/8 个并发请求。实验使用合成的确定性 hot/cold 轨迹，不代表生产到达率分布。

## 结果

`E2E change = 1 - paired/full`，正数表示 paired 更快；负数表示 paired 更慢。p95 是该阶段请求级 p95，缓存计数来自 paired 服务的阶段前后差值。

| 阶段 | 请求数 | Full E2E 均值（ms） | Paired E2E 均值（ms） | E2E 变化 | Full p95（ms） | Paired p95（ms） | Paired target 命中 token | metadata eviction | partial recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 8 | 8019.07 | 8408.35 | -4.85% | 9704.31 | 10442.91 | 40 | 0 | 0 |
| prime | 4 | 2614.51 | 2942.82 | -12.56% | 2888.29 | 3197.43 | 28 | 0 | 0 |
| warm | 8 | 3989.42 | 5618.62 | -40.84% | 5371.09 | 6990.69 | 4088 | 0 | 0 |
| evict | 32 | 11438.03 | 11112.47 | **+2.85%** | 20317.72 | 19766.11 | 317 | 70 | 0 |
| rehit | 8 | 3816.99 | 3744.70 | **+1.89%** | 5132.52 | 5037.32 | 2072 | 5 | 4 次 / 2016 token |

这组数据有两个需要分开看的结论。

第一，生命周期计数确实发生了预期事件。`warm` 阶段累计命中 4088 个 token；`evict` 阶段产生 70 次 metadata eviction；`rehit` 阶段仍然命中 2072 个 token，同时出现 4 次 partial recovery，共恢复 2016 个 token。也就是说，淘汰没有把缓存状态简单地变成“全有”或“全无”，而是留下了可复用的连续前缀，并对 draft 不完整的部分走了恢复路径。这证明本实验捕获到了成对缓存的真实生命周期，而非只统计一个静态命中率。

第二，单阶段 E2E 不能直接等同于缓存收益。`warm` 阶段 paired 比 full 慢 40.84%，说明在这条高并发、短请求批次的合成轨迹中，服务级调度、sidecar bookkeeping 和两套状态维护的开销超过了被省掉的 prefill 工作。`evict` 和 `rehit` 阶段 paired 分别快 2.85% 和 1.89%，但幅度很小，不能据此声称生产加速。把五个阶段按请求数加权后，full 的请求级 E2E 总和为 503078.98 ms，paired 为 509543.80 ms，paired 整体 **慢 1.29%**；按服务阶段墙钟时间相加，full 为 43.56 s，paired 为 45.61 s。

## 为什么 warm 阶段反而变慢

这个结果并不矛盾。当前实验每阶段同时发出 4 个请求，且热前缀只包含 4 个固定 prompt。`warm` 阶段虽然 target 命中 token 很多，但每个请求仍然要经过 paired metadata 查找、readiness 判断、请求生命周期更新和 DSpark verify；实验还关闭了 CUDA Graph 和 compact scan，剩余 decode/verify 时间在短输出下占比不低。full 对照则没有 sidecar 的维护路径。由此可见，“命中 token 多”是必要条件，不能单独推出 E2E 一定更快；还需要把节省的 prefill 时间与 cache lookup、调度和 recovery 成本放在同一条请求轨迹上比较。

## 对工程设计的启示

- 缓存容量至少应同时看 GPU KV 容量和 paired metadata 预算。本实验在 GPU 仍能运行时就产生了 metadata eviction，说明两者不是同一个瓶颈。
- 不能只保留 `target_hit_tokens`。`target` 命中后，draft 是否连续 ready、缺口大小、是否发生 partial recovery，决定了命中是否能转化成真正的计算节省。
- 热前缀回访可以采用分层策略：先复用连续 ready 前缀；缺失后缀进入 partial recovery；只有没有可安全复用的连续前缀时才 full-prefill。这样可以避免因少量 metadata 淘汰而把整个请求退化成冷启动。
- 对生产调度，下一步应把请求到达时间、前缀长度、热度分布和实际 batch 混合进轨迹，并重复多轮。当前结果只有一个固定合成轨迹、每阶段 4～32 个请求，没有给出稳定的收益区间。

## 结论边界

这项任务验证了：在 DSpark paired cache 中，预热、稳定命中、metadata 淘汰和回访后的 partial recovery 都能在同一次服务运行中观测到；它没有证明 paired cache 在所有生产负载下更快。当前配置下，淘汰压力阶段和回访阶段只有约 2% 的均值收益，五阶段累计反而慢 1.29%。要形成可用于容量规划的结论，还需要真实业务 trace、不同 metadata budget、低/中/高到达率以及至少 5～10 个随机化重复轮次。

## 原始证据

- 远端结果目录：`/root/autodl-fs/dspark-cache-prime-eviction-20261117`
- 本地原始 JSON：[results/cache-prime-eviction-20261117/result.json](results/cache-prime-eviction-20261117/result.json)
- 实验协议：[results/cache-prime-eviction-20261117/protocol.json](results/cache-prime-eviction-20261117/protocol.json)
- 远端阶段摘要：[results/cache-prime-eviction-20261117/remote-report.md](results/cache-prime-eviction-20261117/remote-report.md)

## 任务状态

第四个任务已完成。下一项任务尚未开始。
