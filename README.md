# SGLang / DSpark 推测解码优化与验证

围绕 **Qwen3 14B + DSpark** 的 SGLang 推理服务工程项目，研究动态验证调度、target/draft 成对前缀缓存、浮点数值一致性、CUDA Graph、算子融合和 target 量化。

项目的主线是：先确认真实执行路径和性能瓶颈，再实现局部改动，通过逐 token 正确性检查、真实任务质量检查与无 profiler 的服务级 A/B 判断是否保留。仓库同时保存有效方案、被否决的候选和未解决问题。

> **阅读范围与状态：** `main` 保存早期动态验证源码快照、patch 和原始结果；后续工作在 `exp/*` 分支，未整体合并到 `main`。本首页根据截至 **2026-10-29** 的公开分支和内部文档整理，下面的源码、协议和结果链接固定到对应提交，不能视为全部功能已在默认分支启用。
>
> **名称说明：** 这里的 DSpark 是推测解码方案，主要测量硬件是 RTX PRO 6000 Blackwell Server Edition，**不是 NVIDIA DGX Spark**。

## 1. 项目背景：为什么这样做

自回归生成每次只推进一个 token。低并发解码可能受到权重读取、小 kernel 启动和 CPU 调度限制；推测解码用较小的 draft 提出多个候选，再由 target 批量验证，减少 target 执行轮数。本项目典型配置为 draft block 7、最多 8 个验证位置（含 anchor）。

但“少验证 token”或“少调用 kernel”不一定使服务更快，主要矛盾包括：

| 工程问题 | 为什么不能只看局部指标 | 本项目的处理方向 |
|---|---|---|
| 动态验证调度有额外开销 | budget、ragged layout、pack/scatter 的成本可能抵消少做的计算 | 可观测性、graph token 阈值、不可变代价向量缓存 |
| target cache 命中不等于 draft 可复用 | target KV 存在时，draft 所需状态可能尚未 ready | paired metadata、连续 readiness、失效和 partial recovery |
| 串行与并发输出不同 | 相同输入仍可能因 attention 分区、GEMM shape 和舍入顺序产生差异 | 独立 target oracle、中间张量采集与算子重放 |
| CUDA Graph 捕获成功但没有真正重放 | runtime layout、ordered lengths 或 host metadata 不匹配时会回退 | 精确 geometry guard、host mirror bridge、trace 验证 |
| 低 bit 或融合算子未带来端到端收益 | 后端 dispatch、activation quantization、acceptance 和排队仍有成本 | 运行时后端证明、质量门槛、等容量 ABBA、尾延迟检查 |

本项目修改的是已有 SGLang 的局部运行时和实验工具，不是重新实现 SGLang、Qwen 或 DSpark 原始算法。

## 2. 先看哪些分支与文档

早期 `main` 的源码根目录为 `sglang/python/`，后期实验分支为 `python/`。历史文档中的“main/基线”有时指实验谱系根 `0c1f1eb`，不等同于当前 GitHub 默认分支。请按固定提交理解阶段状态。

| 方向 | 分支入口 | 推荐文档或源码 |
|---|---|---|
| 早期动态验证 | [`main` 的原始快照][main-snapshot] | [完整增量 patch][dynamic-patch]、[ABBA 记录][early-results] |
| 后期 FP8 与 cache 总结 | `exp/quant-w8a8-fp8-native-target-bf16-draft` | [阶段更新][project-update]、[优化总结][optimization-update]、[优化历史][history] |
| paired cache 与长输出验证 | `exp/paired-cache-long-validation` | [长输出与 partial recovery 协议][cache-validation] |
| 数值基线 | `exp/numerically-stable-fa4-fixed64` | [稳定基线结果][stable-results]、[batch 诊断][batch-diagnostics] |
| exact target graph | `exp/exact-target-graph8-host-mirror` | [精确 1×8 graph 结果][graph-result] |
| 手写投影融合探索 | `exp/manual-fused-kv-gemm` | [独立 Triton 实验源码][manual-projection] |
| NVFP4 / cuDNN | `exp/quant-nvfp4-cudnn-target-bf16-draft` | [bounded 结果][nvfp4-bounded]、[真实任务停止结果][nvfp4-cudnn] |
| 未解决的长停顿 | `exp/baseline-stall-diagnosis` | [约 60 秒停顿诊断][stall-result] |

其他实验分支包括 compact index 复用、serial reference plan、固定 verify-all、RMSNorm 融合和 norm graph catalog；它们的结果与回退点见[优化历史][history]。不同分支不是一套可任意叠加的生产配置。

## 3. 做了什么：执行链路与关键改动

```text
请求 → Scheduler / target RadixCache → paired readiness 检查
     → 对缺失后缀做 target prefill，并注入 draft 所需 KV
     → draft proposal → budget / verify layout → target verify
     → acceptance finalize → commit / KV 更新 → 输出
```

### 3.1 动态验证调度与可观测性

早期 [patch][dynamic-patch] 的实际增量为 **7 个文件，+535 / -18 行**，包含源码和测试。不是把快照中完整的上游文件都算成新增实现。

- `dspark_observability.py`：增加候选 graph tokens、完整逻辑 tokens、最终路径、planning skipped 和 graph 状态等结构化记录；worker 提交本轮选择与执行信息。
- `dspark_planner.py`：以实际 padding/graph tier 后的 token 数判断 compact 是否值得；节省不足时使用 full-width ragged layout，保留已有 ragged graph executor 的兼容性，而不是直接切回可能只能 eager 的 non-compact 路径。
- 对可以提前判定不划算的情况跳过动态布局构造；按 batch size 与 DP global request tier 复用 uniform layout。
- `HostConfidenceBudgetPlanner`：按请求数缓存 SPS additive step-time 向量，本轮只取候选范围的前缀，避免重复构造插值输入与代价张量。其前提是 planner 生命周期内性能表和相关配置保持不变。
- 在已有运行时控制接口上补充可选 override 的 `null` 清除支持，使固定/自动预算对照可恢复；并非重新发明整个运行时控制 API。

阈值在 planner 初始化时读取，修改环境变量通常需要重建实例。观测中的 pack/scatter 次数由 `run_compact` 推导，不是每个 GPU 调用点的独立计数，精确调用次数仍需 trace。

### 3.2 target / draft 成对前缀缓存

[`dspark_prefix_cache.py`][paired-source] 用 metadata sidecar 绑定 target radix node 与 draft readiness，**不额外复制一套 KV tensor**：target 继续拥有物理 token location，draft 在自己的池中使用对应索引。

实现覆盖：

- 连续 ready prefix 和 fingerprint：不能因为 target 命中就认为 draft 也可用。
- node split/delete、指纹失效、LRU 和 metadata token 预算：让缓存状态随生命周期更新。
- shadow 模式只统计潜在命中，enabled 模式才约束实际复用。
- partial recovery：例如 target 命中 8000、draft 只有前 6000 ready，就保护已有 6000，对缺失后缀重新 prefill 并重建 draft 所需 KV。
- 将 recovery 计划计数与 worker 的实际 prefill/injection 工作量分开观测，避免“计划执行”被误当成“已经执行”。

readiness 依赖已有 CUDA stream 顺序和请求生命周期，不是任意跨 stream 的全局同步保证。详见[实现][paired-source]与[验证协议][cache-validation]。

### 3.3 浮点数值一致性诊断

缓存 A/B 之前，普通 full-prefill 对照就出现了串行与并发 token 差异。因此先建立独立普通 target oracle，再固定输入、输出 token 与 acceptance metadata，采集中间 hidden、Q/K/V、历史 KV 和 attention 输出进行重放。

定位出的案例不是同一个根因：

1. **FA2 分区：** 同一组 Q/K/V 下，B1 与 B8 的 automatic KV chunk 为 128 与 412，attention 输出有 1955/5120 个 BF16 元素不同；保存张量重放可复现。[诊断与 reference plan 验证][reference-plan]
2. **QKV GEMM shape：** 另一案例输入相同，但实际 M8/M32 投影输出有 3022/7168 个元素不同。关闭 reduced-precision reduction 虽减少差异，却也改变原 B1 token，因此该候选被拒绝。[QKV 舍入诊断][qkv-rounding]
3. **FA4 tile：** decode 选择 64×64、verify 选择 64×128，带来 attention 差异。通过进程内 host policy 在首次 JIT 前固定现有 tile 为 64×64，同时保留 target causal 与 draft non-causal 语义，建立独立数值配置。[稳定基线结果][stable-results]

这是对现有 kernel 选择策略和服务正确性的诊断、修复与验证，不是重写了 FlashAttention kernel。32-token 输出的 12 格检查中，普通 target、DSpark full、DSpark paired 各保存 416 个请求通过，合计 1248；128-token 输出另建独立参考。

### 3.4 CUDA Graph 与手写算子

- **精确 graph admission：** 总 token 数相同不代表请求布局相同。capture/runtime 的 batch size、ordered verify lengths、live total 和相关模式需匹配，否则回退 eager。
- **exact 1×8 graph：** 通过 host mirror bridge 补齐当前 device length 的 CPU metadata，再使用原 guard。诊断中观察到 80 次实际 decode 对应 80 次 `cudaGraphLaunch`；D2H 同步成本计入无 profiler A/B。[结果][graph-result]
- **Triton KV 融合：** [`fused_kv_write.py`][kv-source] 融合 K RMSNorm、RoPE 与 KV pool 写回，用 `commit_lens` 屏蔽未提交候选，保留 BF16 中间舍入语义。
- **手写 CUDA compact scan：** [`dspark_compact_scan.cu`][scan-source] 在单个 256-thread CTA 中由线程 0 构建 shared prefix sum，其他线程二分映射 compact row，并 gather token。默认 opt-in，加载失败可回退已有路径；小 batch 设计的扩展性与接口分配开销仍需实测。
- **Triton greedy finalize：** [`dspark_accept.py`][accept-source] 融合首个 mismatch、接受长度、bonus token 和提交 metadata；target logits 的 argmax 仍在外部，不能称为融合了整个 vocabulary sampling。

手写投影分支采用 Triton 乘法归约融合投影与 KV 写回，未实现 Tensor Core GEMM 主循环，也没有已证明的服务级性能优势。CUTLASS epilogue 探针则保留库 mainloop；两者不能混为“手写 CUDA GEMM 已替代 CUTLASS”。

### 3.5 target 量化与运行时证明

FP8 使用已有 `w8a8_fp8` 路径与 `sgl_kernel.gemm.fp8_scaled_mm`，对 target 的 160 个 fused linear 使用 per-output-channel FP8 权重和动态 per-token FP8 activation；embedding、LM head、draft 与 KV 保持 BF16。[recipe][fp8-recipe]

实验不只检查启动参数，还在真实 forward 后检查层类型、weight dtype、转置形状、FP32 scale、storage 和实际后端，排除“参数写 FP8、实际跑 BF16”的无效对照。FP8 成果是配置接入、运行时证明和质量/性能验证，**不是新写 FP8 GEMM**。

NVFP4 另走固定离线 W4A4 checkpoint，并尝试 CUTLASS/cuDNN 后端。bounded 合成请求通过，但真实长数学任务未在固定期限内完成，不能据此宣称 FP4 的稳态收益或质量已通过。

## 4. 取得了什么效果

以下区分“公开机器数据可直接重算”与“仓库历史实验报告记录”。后者不是本次重新连接 GPU 复现的结果。`pN` 表示输入长度，`cN` 表示客户端并发；客户端并发不等于实际 GPU batch size。

| 工作项 | 已记录效果 | 条件与边界 |
|---|---|---|
| planner CPU 微路径 | 90.322 → 39.205 μs，减少约 **56.6%** | 早期历史微基准；不是整个推理服务快 56.6% |
| early dynamic vs fixed | 公开四个 arm 汇总，B 的 mean E2E 比 A 高约 **1.31%** | 每组实际 64 请求；换序结果一快一慢，不证明稳定提速 |
| exact target graph | mean E2E 降低 **1.708%～1.992%** | BF16、p2048/c1、128 输出、精确 1×8；高并发收益未证明 |
| native FP8 target | mean E2E 降低 **31.40%～31.66%** | p2048/c1、128 输出、等 16384 KV 容量，cache/graph 关闭，ABBA |
| FP8 设备显存 | 峰值 **36177 → 24129 MiB**，约减少 **33.3%** | 对应上一行实验；设备级采样，不是纯权重或 allocator 峰值 |
| FP8 较高并发矩阵 | p512～p2048 的合法组合约降低 **10%～25%**；8K c4/c8 约 **31%～32%** | 分容量 suite、AB/BA；不是生产吞吐上限或真实流量回放 |
| paired cache 直接 on/off | BF16 E2E 降低 **20.9%～24.8%**；FP8 降低 **15.0%～16.8%** | p512/c4、32 measured requests、100% 热前缀、特定预热/容量配置 |
| 另一套复用率扫描 | 0/25/50/100% 完整前缀复用时，E2E 分别降低 **1.20% / -0.05% / 1.69% / 2.37%** | BF16 p512/c4、16 measured requests、2 warmup batches；固定顺序，不能与上一行拼接 |

结果入口：[FP8 原生结果][fp8-result]、[FP8 较高并发矩阵][fp8-concurrency]、[cache/graph 结果][cache-result]、[前缀复用率扫描][cache-sweep]。

注意：

- FP8 的 p2048/c1 实验中 acceptance 从 BF16 32.97% 变为 FP8 40.70%，E2E 变化包含接受行为变化，不能全部归因于 GEMM 变快。
- cache 与 FP8 节省的工作有重叠，收益不能相加。cache 开启后，8 个合法组合中只有 3 个同时满足两个顺序 E2E 降低超过 1% 且三项 p95 不退化。
- 100% prefix hit 不保证 E2E 大幅改善，decode、verify 和排队仍可能占主导。不同预热、输出长度和到达协议会改变结果。

### 4.1 FP8 模型质量

使用固定的 384 题子集，冻结输入、prompt、seed、停止条件和 grader，并分别验证同 recipe 的 ordinary AR 与 DSpark 输出。观察分数如下：

| 固定任务子集 | BF16 正确数 | FP8 正确数 |
|---|---:|---:|
| GSM8K | 113/128 | 119/128 |
| 标量 MATH | 25/64 | 26/64 |
| ARC generative | 118/128 | 121/128 |
| MBPP | 52/64 | 54/64 |

这只表明固定子集通过当次质量门槛，不是完整 leaderboard、统计非劣效证明或“量化普遍提高准确率”。MATH 输出上限为 4096，BF16/FP8 分别有 21/20 个截断；ARC 使用生成式答案，不是 likelihood 分数。MBPP 生成代码在非 root、只读 chroot、seccomp 和资源限制下评估。[质量协议][quality-protocol]

### 4.2 早期 ABBA 请求数勘误

保留历史目录名 [`results/abba-4x256/`][early-results]，但其中 `A1/B1/B2/A2.jsonl` **各只有一个聚合记录，`completed=64`，日志的 `num_prompts` 也为 64**。原 README 的“4×256 requests”不符合已公开数据；`random_output_len=256` 是输出长度配置，不能当作请求数。
results/semantic-ab-20261029/  semantic AR/full/dynamic strict comparison and raw logs

| Arm | completed | mean E2E ms | E2E p95 ms |
|---|---:|---:|---:|
| A1 | 64 | 1454.601 | 3380.625 |
| B1 | 64 | 1511.132 | 3505.177 |
| B2 | 64 | 1438.685 | 3265.256 |
| A2 | 64 | 1456.952 | 3382.901 |

两次均值汇总约有 1.31% 差异，但 B1 慢于 A1、B2 快于 A2，存在顺序/波动影响。历史记录称 91 项测试及固定输出检查通过；这些聚合 JSONL 不包含逐请求 token IDs，因此不能据此重新验证全部输出一致性。仓库组装时没有可用 pytest，保留的是测试文件和历史结果，不是新一轮 91-test 复跑。


## 4.3 Semantic strict A/B comparison (2026-10-29)

We ran ordinary AR, DSpark full verify, and DSpark dynamic verify with the same 12 fixed prompts (4 code, 4 math, 4 open QA), the same 512-token input shape, 128 output tokens, deterministic 2 req/s arrival, and c=1/4/8. Cache/radix, FP8, and CUDA Graph were disabled to isolate verifier/planner behavior. Each cell has 12 requests.

The complete raw bundle, runner, server logs, and category table are in [`results/semantic-ab-20261029/`](results/semantic-ab-20261029/). The headline results below are mean/p95 in milliseconds:

| path | c | acceptance | planner CPU | GPU step | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR | 1 | N/A | N/A | N/A | 34328.1/61853.1 | 51.45/53.99 | 40862.0/68338.6 |
| AR | 4 | N/A | N/A | N/A | 4811.8/9460.8 | 51.00/51.58 | 11289.3/15928.1 |
| AR | 8 | N/A | N/A | N/A | 1076.7/2982.0 | 51.69/52.64 | 7641.2/9410.3 |
| full | 1 | 1.000 | N/A | 59.94/63.12 | 3330.7/5889.9 | 7.35/7.65 | 4264.6/6814.5 |
| full | 4 | 1.000 | N/A | 58.17/63.16 | 802.7/1617.8 | 8.93/8.32 | 1936.9/2991.3 |
| full | 8 | 1.000 | N/A | 59.13/63.22 | 580.2/1677.0 | 9.40/8.98 | 1774.6/3114.9 |
| dynamic | 1 | 1.000 | 0.176/0.201 | 63.97/65.28 | 3744.1/6669.7 | 8.06/8.19 | 4767.2/7647.1 |
| dynamic | 4 | 1.000 | 0.189/0.232 | 60.94/64.97 | 834.8/1612.6 | 9.53/8.95 | 2045.1/3080.0 |
| dynamic | 8 | 1.000 | 0.186/0.254 | 60.63/65.08 | 547.1/1627.1 | 9.68/9.20 | 1776.7/3094.0 |

This run shows the expected decode-throughput advantage of speculative verification. Dynamic planning is slightly worse than static full verify at c=1 and c=4 because compact planning/layout overhead is visible; at c=8 it is effectively tied with full verify. Acceptance was 1.0 for every semantic category, so this fixed padded prompt set is a systems-path baseline, not evidence of answer-quality improvement. No exact-match, code unit-test, or LLM-judge score was collected in this run; see `RESULTS.md` for the required next quality loop and all limitations.
## 5. 没有成功或尚未解决的路线

| 路线 | 观察 | 决策 |
|---|---|---|
| uniform layout cache 作为主要瓶颈假设 | 命中超过 99%，节省仅约 19 μs/step | 不作为主要服务收益来源 |
| 固定 verify-all | mean E2E 降低约 0.515%，BA 未完成 | 未达到预设 1% 收益门槛 |
| residual RMSNorm 点运算融合 | norm kernel 990 → 270；E2E 只降低 0.272%，TTFT p95 增加 4.002% | 否决；kernel 数减少不等于服务变快 |
| 原算术 norm graph catalog | E2E 仅降低 0.0634%，E2E p95 增加 5.807%，设备采样显存增加 5706 MiB | 否决 |
| NVFP4 CUTLASS / cuDNN | 112 个 bounded 合成请求通过，真实长任务在 240 秒期限内失败 | 未完成质量和稳态收益资格验证，不宣称成功 |
| graph 较高并发 | baseline noise/warmup gate 或约 60 秒停顿阻止候选计时 | 无候选较高并发提速结论 |
| 约 60 秒停顿 | baseline 中也发生，graph-off 仍存在 | 根因未确认；I/O、JIT、GC、CPU/驱动等待只是待验证假设 |
| 手写投影融合 / CUTLASS epilogue / top-k-first sampling | 有探索代码，缺少独立稳定的服务收益证明 | 保留为实验，不包装为已落地加速 |

失败协议与停止原因见[优化历史][history]、[RMSNorm A/B][norm-result]、[norm graph 结果][norm-graph-result]和[停顿诊断][stall-result]。局部 kernel 时间、GPU busy union、CPU NVTX wall、服务 E2E 是不同指标，不互相替代。

## 6. 技术栈与贡献边界

| 技术 | 在项目中的用途 |
|---|---|
| SGLang SRT、Scheduler、RadixCache、KV pool | 推理执行、调度、缓存与 DSpark 接入 |
| Qwen3 14B、DSpark draft | target/draft 验证负载；使用上游模型与算法 |
| Python、PyTorch | planner、状态管理、张量参考、运行时 hook 和实验工具 |
| Triton | KV norm/RoPE/write、greedy finalize、norm 与投影融合实验 |
| CUDA C++、PyTorch extension | 手写 compact scan 与 token gather；不是成熟 GEMM 替代品 |
| FlashInfer FA2/FA4、CUDA Graph | attention 分区诊断、tile policy、capture/replay 与安全回退 |
| 原生 W8A8 FP8、ModelOpt NVFP4、CUTLASS、cuDNN | 量化配置和已有计算后端的接入与比较 |
| asyncio、aiohttp、SSE | 并发服务请求、首 token/输出计时和中断 |
| Nsight Systems、NVTX、SQLite、NVML | CPU/GPU 关联、kernel/graph 路径归因与设备显存采样 |
| unittest/pytest、Git、SHA256、JSONL | 回归、固定输入与证据完整性；测试保留不代表均已重新运行 |
| chroot、seccomp、资源限制 | 生成代码质量评估的隔离执行 |

主要硬件记录为 RTX PRO 6000 Blackwell Server Edition（SM120，约 96 GB 显存）、TP1、Qwen3 14B 与 5 层 draft。10 月报告还记录了驱动 595.71.05、Torch 2.13/cu130、FlashInfer 0.6.17；这是当时环境记录，**不是完整可安装的依赖锁文件**。各阶段环境不同，绝对时间不能跨机器直接比较。

## 7. 仓库结构与复现入口

`main` 当前保存：

```text
sglang/python/sglang/srt/      早期改动涉及的源码快照
sglang/test/                 对应回归测试
sglang-dspark-dynamic-verification.patch
results/sps-probe/           SPS probe 数据、manifest 与日志
results/abba-4x256/          历史 ABBA 聚合数据与日志（实际每组 64 请求）
SHA256SUMS                  所列文件的完整性校验
```

后续 `exp/*` 分支还包含 `python/sglang/`、`benchmark/dspark/` 和 `docs/`，须使用相应分支/固定提交的文档入口，不能假定这些目录已存在于 `main`。

### 7.1 使用早期 patch

本仓库是**增量 patch、部分源码快照和实验材料集合**，不是完整、可独立安装的 SGLang 发行版。需要先准备匹配版本的完整上游 checkout，再检查 patch：

```bash
git clone https://github.com/Nemurizz/SGLang-DSpark-.git
cd /path/to/compatible/sglang
git apply --check /path/to/SGLang-DSpark-/sglang-dspark-dynamic-verification.patch
git apply /path/to/SGLang-DSpark-/sglang-dspark-dynamic-verification.patch
```

不要把早期 patch 直接应用到任意最新版或后期实验目录。`git apply --check` 通过也不代表整个服务功能、kernel 编译和性能验证通过。

可在 LF 检出的仓库根目录验证公开文件：

```bash
sha256sum -c SHA256SUMS
```

Windows 的自动 CRLF 转换会改变字节，校验时需使用原始 LF/Git blob 内容。校验只验证文件完整性，不验证实验结论。

### 7.2 后期实验的验证顺序

1. 补齐完整上游 checkout、model/draft revision、SPS table、依赖版本、GPU/driver 和该分支的数值 policy。
2. 做 backend smoke 和运行时 dtype/storage/backend 检查。
3. 建立独立普通 target oracle；同 recipe 的 DSpark 对照 token 与 acceptance metadata。
4. 量化另做真实任务质量门槛，不能要求 FP8 输出必须等于 BF16，也不能把不同 token 自动判为执行 bug。
5. baseline 先通过噪声/预热门槛，再做不带 profiler 的 ABBA，固定输入/输出、请求数、KV 容量、并发、停止条件，并检查 TTFT/TPOT/E2E 及其 p95。
6. profiler 仅用于机制解释；带张量 D2H、observer 或 nsys 的计时不用于宣称最终性能。

TTFT 取首个带 token 的 SSE frame；TPOT 是首末 token 到达间隔除以输出数减一，是 speculative burst 下的摊销指标，不是人为拆分出来的逐 token ITL。

## 8. 公开证据与当前限制

- 2026-10-16 的全仓库核对覆盖当时 27 个公开分支、41 个 tag、117 个可达提交；分支/tag tip 文件树去重后为 212 个路径、285 个文件版本，含 39 个 Markdown 路径、49 个版本。这是本次 README 更新前的覆盖口径，不是持续自动更新的计数。
- 早期 `main` 的 36 项校验均可按 Git blob 原始字节核对。后期许多 raw requests、trace 和运行证明仅在报告中列出服务器路径。
- [FP8 结果][fp8-result]链接的 `outputs/dspark-quant-fp8/dspark-quant-fp8-quality-20261009-h66-v2.tar.gz` 未出现在核对过的公开分支/tag 文件树中。因此后期数字应理解为**仓库实验报告记录**，不是公开原始材料已全部齐备的独立复现。
- 部分完整上游文件首次被 Git 跟踪时，会显示为数千行新增；这不代表这些代码由本项目全部编写。历史运行目录还依赖 8357 个原有未跟踪路径，checkout 某个实验分支并不自动得到完整运行环境。
- 本次文档核对包括公开 JSON 重算、源码/协议交叉检查和 205 个 Python 文件版本的 AST 语法检查；没有重新连接 GPU 跑 benchmark。语法通过不代表 kernel 编译、正确性、性能或生产资格通过。
- 尚未提供真实生产流量回放、长周期稳定性、多机扩展或全负载收益证明。下一步优先补齐原始证据归档、完整依赖/模型锁定文件和真实前缀复用分布测试。

## Upstream / License

基于 [SGLang](https://github.com/sgl-project/sglang) 及其已有推理、缓存、attention 和量化生态。Qwen、DSpark 原始方案与已有库 kernel 的贡献归原作者。本仓库保留上游许可，见 [LICENSE](LICENSE)。

[main-snapshot]: https://github.com/Nemurizz/SGLang-DSpark-/tree/2ec9ac3025ba829f639dead6808f655e7ef166d1
[dynamic-patch]: https://github.com/Nemurizz/SGLang-DSpark-/blob/2ec9ac3025ba829f639dead6808f655e7ef166d1/sglang-dspark-dynamic-verification.patch
[early-results]: https://github.com/Nemurizz/SGLang-DSpark-/tree/2ec9ac3025ba829f639dead6808f655e7ef166d1/results/abba-4x256
[project-update]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/docs/PROJECT_UPDATE_20261009_FP8_PAIRED_CACHE.md
[optimization-update]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/DSPARK_OPTIMIZATION_UPDATE_20261009.md
[history]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/docs/DSPARK_OPTIMIZATION_HISTORY.md
[cache-validation]: https://github.com/Nemurizz/SGLang-DSpark-/blob/fc37581749a12d0bf466194914602da30af8ec79/benchmark/dspark/STABLE_CACHE_VALIDATION.md
[stable-results]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/STABLE_BASELINE_RESULTS_20261007.md
[batch-diagnostics]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/BATCH_DIAGNOSTICS.md
[graph-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/80aafad21d166b6f41b8818639741018bc748e3e/benchmark/dspark/EXACT_TARGET_GRAPH8.md
[manual-projection]: https://github.com/Nemurizz/SGLang-DSpark-/blob/6c7fb1398e9e7c1051cb17841b15bd3f578aa5a1/python/sglang/kernels/ops/speculative/dspark/fused_kv_write.py
[nvfp4-bounded]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/NVFP4_TARGET_BOUNDED_RESULT.md
[nvfp4-cudnn]: https://github.com/Nemurizz/SGLang-DSpark-/blob/146324709e913ded16fe6bcb62e1df9bf835dcc2/benchmark/dspark/NVFP4_CUDNN_EXECUTION_RESULT.md
[stall-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/d6444fcf5cf107726949111ee82b0d067e91806d/benchmark/dspark/BASELINE_STALL_DIAGNOSIS.md
[paired-source]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/python/sglang/srt/mem_cache/dspark_prefix_cache.py
[reference-plan]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/REFERENCE_PLAN_VALIDATION_20261007.md
[qkv-rounding]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/BASELINE_BATCH_ROUNDING_20261007.md
[kv-source]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/python/sglang/kernels/ops/speculative/dspark/fused_kv_write.py
[scan-source]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/python/sglang/kernels/ops/speculative/dspark/dspark_compact_scan.cu
[accept-source]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py
[fp8-recipe]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/fp8_recipe.py
[fp8-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/FP8_NATIVE_RESULT.md
[fp8-concurrency]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/FP8_HIGH_CONC_RESULT_20261009.md
[cache-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/FP8_DSPARK_CACHE_GRAPH_RESULT_20261009.md
[cache-sweep]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/CACHE_HIT_RATE_SWEEP_RESULT_20261009.md
[quality-protocol]: https://github.com/Nemurizz/SGLang-DSpark-/blob/b946561934fcdf9143aa6e4e814b6c35588291ba/benchmark/dspark/NVFP4_QUALITY_PROTOCOL.md
[norm-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/e70cfd6097618929084a7260a9185d32cc07b6cf/docs/DSPARK_RESIDUAL_NORM_AB_20261008.md
[norm-graph-result]: https://github.com/Nemurizz/SGLang-DSpark-/blob/9d90c615374cf4083de5306c76cdd4fffdbf01a2/benchmark/dspark/ORIGINAL_NORM_GRAPH.md

## CPU planner supplementary measurement (2026-10-29)

This supplement measures only the DSpark CPU planner and its real compact-verify serving path. It does not claim a cache, FP8, CUDA Graph, or semantic-quality improvement. The run used Qwen3-14B with `dspark_qwen3_14b_block7` on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

- Target and KV cache used BF16; FP8, prefix/radix cache, CUDA Graph, compact CUDA scan, and simulated acceptance were disabled. The service ran with `SGLANG_RAGGED_VERIFY_MODE=compact` and `SGLANG_SIMULATE_ACC_LEN=-1`, so acceptance lengths came from real execution.
- The serving matrix used random token IDs for hardware/planner shape measurement, input lengths 512/2048/8192, concurrency 1/4/8/16 where the 131072-token capacity allowed it, and output length 128. `p8192/c16` was recorded as capacity-skipped because the prompt alone fills the configured token capacity.
- At concurrency 1, the observed planner selected budget 7 and verify length 8 throughout these cells. At higher concurrency, budgets formed different steps (for example, up to 112 at p2048/c16), and the records captured the corresponding compact/full token counts and accepted lengths.
- The isolated additive-cost-vector microbenchmark measured uncached median planner time of about 84.3–94.2 us versus 38.7–42.6 us with the per-request-count cache, about 2.18–2.21x, with identical budget decisions. This is a CPU micro-path result, not an end-to-end speedup claim.
- `step_cpu_ms` in the serving records is the full decode-step CPU interval. The supplement does not label it as planner-exclusive time. `predicted_step_ms` comes from the existing SPS table and must not be compared directly with the BF16/no-Graph target GPU time.

The complete protocol, raw per-cell records, server configuration, and rerun scripts are in [`results/cpu-planner-retest-20261029/`](results/cpu-planner-retest-20261029/RESULTS.md).
