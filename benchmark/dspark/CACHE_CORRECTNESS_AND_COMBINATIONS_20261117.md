# DSpark 第 5、6 项：补充正确性与组合开关 AB/BA（2026-11-17）

## 第 5 项：补充正确性场景

第 5 项没有把“调用过某个函数”当成正确性通过，而是针对 sidecar 的生命周期边界做了六组独立测试。测试直接调用当前 `DsparkPrefixCache` 实现，并使用 fake radix node 构造 split、LRU 和 fingerprint 场景；第 4 项的真实 GPU `p512/c4` 流程又覆盖了并发同前缀、mixed hit/miss、eviction 后回访和 partial recovery。

| 场景 | 检查内容 | 结果 |
|---|---|---|
| simultaneous same-prefix fill | 两个线程同时把同一 prefix 标记为 ready，随后读取连续 paired prefix | **PASS**，hit length 8，无异常 |
| split/read race | 一个线程反复读取旧 child，另一个线程执行 node split | **PASS**，500 次读取均返回非负有效长度 |
| cancellation boundary | 模拟请求取消前没有完成 draft KV 写入，确认 readiness 不会被提前保留 | **PASS**，完成前 hit 0，成功写入后 hit 8 |
| LRU eviction during request | 极小 metadata budget 下插入、touch、再插入，观察 eviction 与 readiness | **PASS**，发生 3 次 eviction，旧状态没有继续伪装成 ready |
| fingerprint invalidation | 从 `fp-a` 切换到 `fp-b` 后读取旧节点 | **PASS**，旧 hit 8，新 fingerprint hit 0，tracked token 清零 |
| async stream not ready | `ready=False` 时不应命中，完成后 `ready=True` 才允许命中 | **PASS**，未完成 hit 0，完成后 hit 8 |

这六组测试证明了 sidecar 数据结构在这些边界上的预期状态转换。取消测试目前是 readiness 边界模拟，不是 HTTP 请求在 CUDA worker 正在执行时的真实 abort；异步 stream 测试也验证的是 worker 只有在完成后调用 `mark_inserted_path(ready=True)` 的契约，不是对 CUDA event 的硬件级 timing 验证。这两项仍需要真实请求取消和 stream event trace 才能扩大结论范围。

## 第 6 项：组合开关独立 AB/BA 冒烟

每个组合都固定 paired cache 开启，执行 `OFF → ON → ON → OFF` 四个独立服务 arm；每个 arm 先用 4 个共享前缀请求 prime，再用同一组 4 个请求计时。每个成功 arm 的两个重复运行逐 token 和五项 acceptance metadata 一致。每个 arm 只有 4 个 measured requests，因此这些数据是组合可用性和方向性 smoke，不是 64-request 稳态收益证明。

`E2E gain = 1 - ON/OFF`。AB 使用 arm0→arm1，BA 使用 arm3→arm2。正数表示 ON 更快。

| 组合 | AB E2E gain | BA E2E gain | ON correctness | 结果 |
|---|---:|---:|---|---|
| paired cache + native FP8 | +4.15% | +4.28% | PASS | 通过 smoke；ON 峰值显存 26907 MiB，BF16 paired 约 38955 MiB |
| paired cache + compact scan | -5.46% | +2.38% | PASS | 方向不稳定；AB 的 E2E p95 ratio 1.052，不能宣称收益 |
| paired cache + adaptive draft | 修复后完成 smoke | 修复后完成 smoke | PASS（修复前失败） | planner 缓存按当前 candidate pool 长度扩容；修复后 8 请求完成，cached tokens=3529 |
| paired cache + CUDA Graph | +31.54% | +25.40% | PASS | graph smoke 通过；ON 峰值约 42257 MiB，但它移除了 fixed FA4 hook 和 deterministic flag，不是纯 graph-only 隔离收益 |
| paired cache + KV quantization | 修复后完成 smoke | 修复后完成 smoke | PASS（源码自动路由） | FP8 KV + FA4 自动切到 FlashInfer；8 请求完成，cached tokens=3529；deterministic+FA4+FP8 改为启动期可操作报错 |
| paired cache + long/high saturation（p2048/c4） | -1.95% | +5.92% | PASS | 同一组合方向不稳定，不能宣称长上下文稳定加速 |

补充尾延迟结果：FP8 AB/BA 的 E2E p95 ratio 为 0.984/0.992；compact scan 为 1.052/0.977；CUDA Graph 为 0.696/0.761；long/high saturation 为 1.019/0.940。compact scan 的 AB 尾延迟已退化，long/high saturation 的 AB 均值和 p95 也没有通过稳定收益门槛。

## 怎么解释这些组合结果

FP8 是本轮最干净的成功组合：paired cache 命中保持在 2044 token，两个顺序都降低 E2E，且重复 arm 的输出与 acceptance 一致。它仍然只是一个 4-request smoke，不能替代已有的高并发 FP8 ABBA 矩阵。

compact scan 的功能路径能跑通，但收益受 request geometry 和 launch/synchronization 成本影响，AB/BA 方向相反；这正是不能把“手写 CUDA kernel 能加载”写成“服务加速”的原因。

adaptive draft 的故障来自按 request 数缓存 planner cost vector，candidate pool 在 adaptive 调整后变长。现已在每次命中缓存时检查当前所需长度并扩容；修复后服务 smoke 通过。后续再测性能时仍要做多轮 ABBA，不能把兼容性 smoke 当成收益证明。

KV quantization 的失败发生在 FA4 attention 输入 dtype 校验：Q 仍是 BF16，而 KV 变成 FP8。现已在 server_args 的兼容阶段把非 deterministic 的 FP8+FA4 组合自动路由到支持 FP8 KV dequant 的 FlashInfer，并对 deterministic+FA4+FP8 提前给出明确错误；没有吞掉 kernel assertion。

CUDA Graph smoke 的 ON arm 确实成功运行，但为了绕开当前 fixed FA4 profile 对“deterministic + graph off”的强制约束，测试移除了 fixed hook 和 deterministic flag。因此它证明的是“paired cache + 标准 graph 配置可以运行并在本 smoke 中更快”，不能直接与 fixed64 graph-off 基线相加，也不能当成严格的单变量 graph gain。

## 结论

- 第 5 项六类 sidecar 生命周期边界测试全部通过；真实 HTTP abort 和 CUDA event 级 stream 证据仍未覆盖。
- 第 6 项中，FP8 组合通过当前 smoke；compact scan 和长上下文/高负载方向不稳定；adaptive draft 与 KV quantization 暴露了真实配置/实现错误；CUDA Graph 通过的是放宽 numerical profile 后的独立 smoke。
- 这些结果不支持把六个开关全部同时打开。下一步应先修复 adaptive cost-vector 覆盖和 KV/FA4 dtype 匹配，再分别扩大成功组合的 ABBA 样本；graph 则应建立独立 numerical profile 和 exact geometry correctness gate。

## 原始证据

- 第 5 项 sidecar 测试：[results/cache-supplementary-correctness-20261117.json](results/cache-supplementary-correctness-20261117.json)
- 第 6 项协议：[results/cache-combo-abba-20261117/protocol.json](results/cache-combo-abba-20261117/protocol.json)
- 第 6 项逐 arm 原始结果：[results/cache-combo-abba-20261117/result.json](results/cache-combo-abba-20261117/result.json)

## 任务状态

第 5、6 项均已完成。后续任务尚未开始。
