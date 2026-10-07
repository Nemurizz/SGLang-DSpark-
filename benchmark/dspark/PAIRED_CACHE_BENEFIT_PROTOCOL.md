# paired-cache 提速：固定验收协议

本协议回答“如何证明收益”，不是已经证明收益。用户已明确允许建立原代码 cache-compatible 配置的独立对照，同时保留既有 deterministic 参考。GEMM、graph、量化均不进入此次比较。

## 1. 对照定义：只改变 cache treatment

硬件、模型与 draft 权重、dtype、TP、prefill/decode backend、greedy 参数、seed、graph 状态、draft 策略、HTTP 连接策略、prompt IDs、输出长度、并发及到达方式一致。deterministic 一律关闭，graph 一律关闭，不跨 deterministic ON/OFF 比速度。

新原始参考只能由 clean tracked `a61e0bb` 的 full-prefill 配置生成；不从优化候选重建。首先要求该配置的冷串行、重复串行、并发输出与全部 acceptance metadata 一致。首次不一致立即停服，不能发布完整有效 oracle，不能继续 paired 测速。32-token gate 仅作初步淘汰；正式使用 128/512-token 输出须建立并验证对应长度的原始参考。

| 组 | 作用 | 条件 |
| --- | --- | --- |
| A：full prefill | 服务级性能对照 | DSpark 原 guard 禁止 cache reuse |
| B：paired cache | 待验收方案 | 只复用同 fingerprint、连续 ready 的 target/draft 前缀 |
| B-cold：paired 开启但不命中 | 验证未命中时的额外成本 | 与 A 的同一冷请求对照，不能混入暖命中收益 |
| C：正确的 target-only + draft 恢复 | 区分 paired 的额外收益 | 必须通过相同正确性门槛；当前尚未具备有效结果 |

现有 `target_only_shadow` 会允许不安全的 target-only reuse，不能用错误输出的速度当作有效 C。若 C 不满足正确性，只能证明 B 相对 A 的收益，不能声称 paired 优于正确的 target-only 系统。pure shadow 只证明可复用潜力，不证明实际跳过了计算。

## 2. 正确性必须先过关

- 每个请求逐 token 匹配独立的原始 full 参考，而非只比较文本 hash 或最终答案。
- 同请求比较 verify 次数、correct/proposed drafts、acceptance/correct-draft histograms 五项元数据。
- 冷请求、prime、暖命中、partial recovery、eviction 后重新请求均必须检查；partial recovery 没有事件则标为未覆盖。
- 覆盖 `128/512/2048/8192 × 1/4/8`，并进一步验证正式输出长度和持出 prompt；串行重复一致不等于并发稳定。
- 输出不一致、acceptance 变化、非法访问、scheduler/CUDA 异常或达到样本门槛后的 p95 退化：立即停止本轮服务、回到已知检查点，保留失败记录。

## 3. 证明“确实命中并省了工作”

需要同一测量窗口中的证据：

- shadow target/paired 候选覆盖：潜在命中率和可用连续前缀。
- 实际 `cached_tokens`、实际 paired hit length、target/draft readiness 与 fingerprint：两边都有效，不把 target-only hit 当成 paired hit。
- 实际 target extend/prefill tokens、draft KV 重建或 recovery tokens，证明计算/写入确实减少；可用诊断日志或单独的 nsys trace 确认。profiling 开启的延迟不混入正常计时。
- partial/full recovery 的计划次数、实际执行次数与实际 draft fallback 分开。当前 sidecar recovery counters **仅是计划**，实际 worker fallback 尚未独立暴露，需要补齐后才能交付这一指标。
- 统计 match 重试与请求数；只有分母有效时才报告 request/token hit rate。不知道的值为 unavailable/null，不能填 0。

命中率高不是充分条件：净收益等于跳过的工作，减去 sidecar 查找、恢复、metadata、调度和显存压力等新增成本。paired KV 复用不缩短模型的上下文，所以后续 decode 仍要读取该上下文；TPOT 不一定降低。

## 4. 稳定态 A/B：减少顺序与热身偏差

- 固定 prompt IDs，严格 token 长度；同前缀、不同 suffix 的请求是关键场景。完全相同 prompt 的重复只能作为理想上限，不能替代实际共享前缀负载。
- 分别报告：冷 miss、已 prime 的 warm hit、固定 hit/miss 混合负载。混合比例在测试前锁定，不挑事后最有利的命中率。
- 同样的启动和 warmup，cache prime 成本单列；长期结果须说明 prime 成本是否摊销。模型/JIT warmup 不能和 cache warmup 混为一谈。
- 独立进程轮次按 A/B、B/A 交替或预先随机平衡顺序；每轮 cache 生命周期相同。当前 harness v3 的 full→paired 固定顺序不满足这一完整要求，后续需要扩展。
- 使用独占 GPU，记录温度、功耗、利用率、软件版本和每轮 manifest。条件变化的轮次单独报告，不能偷偷删除慢样本。
- 当前每格 64 样本是 p95 guard 的最低工程门槛，不是充分的统计保证。正式从每格每轮至少 256 请求、多轮独立 A/B 起步；根据 pilot 的波动增加样本和独立轮次，报告轮次级不确定性，不能把同一批相关请求当成独立轮次。
- 正式延迟场景至少使用 128-token 输出；另行测较长输出以辨别 TTFT 改善是否被 decode 摊薄。新长度不能沿用 32-token oracle。

## 5. 指标与通过条件

服务基准通常记录 TTFT、TPOT、端到端延迟、吞吐与 speculative acceptance；SGLang 的定义见 [官方 Bench Serving Guide](https://docs.sglang.ai/developer_guide/bench_serving)。本项目以累计 output_ids 和 token-bearing SSE frame 计时，TPOT 是 burst 输出下的 amortized 平均，不伪造逐 token ITL。

- 主指标：各格 TTFT p50/p95；同时报告 TPOT p50/p95、E2E p95、吞吐与显存。
- `TTFT 降幅 = (A − B) / A × 100%`；`TTFT 加速倍数 = A / B`，不能混用两个数。
- 在相同、预先确定的 workload 下，改善须跨独立轮次可重复，轮次级 95% 置信区间应排除“无改善”；实际最小有用幅度应在正式跑数前确定，不能事后修改以凑成功。
- TPOT、TTFT/E2E p95 不得退化；显存增量、可容纳并发/KV 容量必须报告。任一格失败，不宣称整矩阵通过；有收益的子集也只能作为被完整验证的适用范围，不能隐藏失败格。
- 不把暖命中的 TTFT 降幅当作整体 E2E/吞吐加速。示例（非实测，忽略尾部开销）：TTFT 800→560 ms 降低 30%；若 128-token 输出的 TPOT 都是 20 ms，整体约为 3340→3100 ms，只降低约 7.2%。
- target-only C 缺失时明确结论范围；partial recovery 未覆盖、fallback 未观测、p95 样本或轮次不足时，相关结论标为“未证明”。

## 当前进展和缺口

已具备：固定 12 格数据集、逐 token + 五项 acceptance gate、独立原始参考机制、无 generate 重试、SSE burst 记录、设备级显存采样、失败停服及可回退提交。

尚未完成：cache-compatible 原始 baseline 的整矩阵稳定性验收；有效 paired A/B；平衡顺序调度；实际 worker fallback instrumentation；充分稳定态样本/轮次；冷/暖/混合场景及 partial recovery 压力覆盖。

因此，本项目目前没有 paired-cache 端到端收益数字。先通过正确性，才能测得可以使用的速度数字。

本次执行结果：已按授权尝试原 cache-compatible baseline 独立 gate。`128 × 1` 三轮通过，但 `128 × 4` 首轮并发的 variant 0 第 3 个 token 从独立串行参考 `34334` 变为 `16555`，correct drafts 从 22 变为 21。立即停服；未发布完整新 oracle，未进入 paired。原 deterministic 参考 SHA 未变化。具体结果见同级输出目录中的独立 baseline 稳定性结果。
