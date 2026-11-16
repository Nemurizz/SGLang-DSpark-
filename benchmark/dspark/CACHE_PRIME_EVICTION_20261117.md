# DSpark paired cache：prime / eviction 完整账本（2026-11-17）

这次实验按同一个 `p512/c4` cell 记录完整生命周期：`cold_miss → first_prime → warm_hit → mixed_hit_miss → eviction_pressure → rehit_after_eviction`。full-prefill 和 paired cache 使用同一组 token IDs、同一输出长度 128、同一 FA4 fixed64 profile（`0577109`），独立进程运行。paired 侧临时加入了 Python-side cache instrumentation；服务结束后已恢复原始源码。

除了服务端返回的 TTFT/TPOT/E2E，本轮还保存了：请求级累计 E2E、SGLang `ReqTimeStats` 的 queue wait、sidecar cache CPU、metadata 浅层内存估算、NVML 设备显存、eviction/recovery counters 和逐请求 token/acceptance 对照。metadata memory 是 Python 对象的浅层估算，不能当成 GPU KV 显存；cache CPU 是 sidecar 方法的包含式计时，嵌套方法会被分别计入。

## 逐阶段结果

| 阶段 | N | Full TTFT 均值 ms | Paired TTFT 均值 ms | Full 累计 E2E ms | Paired 累计 E2E ms | Full queue wait 均值 ms | Paired queue wait 均值 ms | Paired cache CPU ms | metadata bytes 前→后 | GPU peak MiB | eviction / recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold_miss | 1 | 2799.39 | 2633.14 | 7150.41 | 6855.15 | 0.20 | 0.25 | 0.225 | 192→1560 | 38687 | 0 / 0 |
| first_prime | 4 | 259.16 | 257.02 | 11108.61 | 10443.47 | 0.23 | 0.64 | 0.319 | 1560→5028 | 38973 | 0 / 0 |
| warm_hit | 4 | 262.69 | 1888.47 | 11018.59 | 16984.79 | 0.25 | 1.07 | 0.363 | 5028→7340 | 38973 | 0 / 0 |
| mixed_hit_miss | 8 | 1718.87 | 1558.86 | 34166.37 | 31443.30 | 1483.25 | 1363.04 | 0.672 | 7340→11654 | 38973 | 0 / 0 |
| eviction_pressure | 32 | 9739.66 | 9153.69 | 391572.03 | 367853.63 | 6108.82 | 5743.73 | 2.941 | 11654→13488 | 38973 | 64 / 0 |
| rehit_after_eviction | 4 | 247.45 | 253.42 | 10625.33 | 10168.43 | 0.19 | 0.76 | 0.313 | 13488→15456 | 38973 | 5 / 4 |

queue wait 来自服务端日志中的 `ReqTimeStats(queue_duration=...)`，并按阶段请求数量切分；启动阶段额外的 6-token 内部请求已剔除。混合和 eviction 阶段的 queue wait 明显上升，说明这两段的 E2E 主要已经受到请求排队影响。

## 这组账本说明了什么

第一，缓存生命周期事件真实发生了。`warm_hit` 阶段 paired 命中 2044 个 token；`eviction_pressure` 阶段发生 64 次 metadata eviction；`rehit_after_eviction` 仍有 2044 个 target-hit token，其中 28 个 token 直接 paired 命中，另外 4 次 partial recovery 共计划恢复 2016 个 token，并再发生 5 次 eviction。每个 paired 请求的输出 token 和五项 acceptance metadata 都通过了 full-prefill 对照。

第二，metadata 和 GPU KV 是两个不同的容量维度。paired metadata 浅层估算从 192 bytes 增长到 15456 bytes，GPU 设备显存峰值约 38.7～39.0 GiB；eviction 发生时 GPU 仍有容量余量。因此只监控 GPU 显存不能判断 draft readiness metadata 是否已经被挤出。

第三，cache CPU 本身很小，但不是零。各阶段 sidecar 方法包含式 CPU 时间约 0.225～2.941 ms；eviction pressure 的增量最大。这个数字是 Python-side instrumentation 的归因结果，包含 wrapper 和嵌套调用开销，适合做数量级账本，不适合当作无插桩服务延迟。

第四，warm hit 没有自动转化成更低的服务 E2E：该阶段 paired 的 TTFT 均值升到 1888.47 ms，而 full 为 262.69 ms。此阶段只有 4 个并发请求、输出短，且 sidecar/readiness 与调度成本在总时间中占比很高。到了 mixed 和 eviction 阶段，paired 的累计 E2E 分别比 full 低 7.97% 和 6.05%，说明高排队压力下省下的 prefill 工作开始抵消部分 cache 管理成本。

六个阶段的请求级累计 E2E 总和为：full **465641.36 ms**，paired **443748.78 ms**，paired 在这条单次合成轨迹上整体快 **4.70%**。这是一个 cell 的描述性结果，不能外推为稳定生产收益；它也没有证明实际 worker fallback 已经等于 sidecar 的 recovery plan。当前代码仍只暴露 plan counters，因此 recovery 的“实际执行次数”应标为未独立测量。

## 需要特别保留的边界

- `cold_miss` 的 sidecar `match_calls=2`，而请求数为 1，说明存在匹配重试；不能把该阶段的 token hit rate 当成有效请求分母。
- 当前只有一个 `p512/c4` cell、一次 full/paired 顺序，没有 ABBA、多轮重复或置信区间；p95 只作描述。
- metadata memory 是浅层 Python 对象估算；GPU memory 是 NVML 设备级峰值，不是 PyTorch allocator peak。
- queue wait 已从 `ReqTimeStats` 日志提取，但没有逐请求 RID 与客户端响应行的严格配对；本报告使用阶段顺序和样本数切分，阶段级统计可用，逐请求因果归因仍需 trace ID。
- mixed、eviction 和 rehit 的结果包含排队效应；不能把 TTFT/E2E 的变化全部归因于 cache lookup 或 prefix hit。

## 下一步

按图片中的任务顺序，下一项应先做补充正确性场景：两个请求同时填充同一前缀、split 节点被另一请求读取、取消时 readiness 保留、LRU eviction 与请求并发、fingerprint 变化失效、以及 target/draft 异步 stream 尚未完成时不能提前标 ready。只有这些场景通过，才进入组合开关 AB/BA。

## 原始证据

- 远端结果目录：`/root/autodl-fs/dspark-cache-prime-eviction-ledger-20261117`
- 本地账本结果：[results/cache-prime-eviction-ledger-20261117/result.json](results/cache-prime-eviction-ledger-20261117/result.json)
- queue wait 阶段统计：[results/cache-prime-eviction-ledger-20261117/queue_wait_from_log.json](results/cache-prime-eviction-ledger-20261117/queue_wait_from_log.json)
- GPU 采样：[full-gpu-samples.jsonl](results/cache-prime-eviction-ledger-20261117/full-gpu-samples.jsonl)、[paired-gpu-samples.jsonl](results/cache-prime-eviction-ledger-20261117/paired-gpu-samples.jsonl)

## 任务状态

图片中第 4 项已按要求补齐账本字段并完成一个 `p512/c4` cell。第 5 项补充正确性场景尚未开始。
