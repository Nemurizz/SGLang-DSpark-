# Residual RMSNorm 融合：结果与止损（2026-10-08）

## 结论

**候选不合格，已经停止并撤销进程内 hook，未合并、未默认启用。≥1% 端到端收益未证明。**

独立实验只融合 residual RMSNorm 的点运算，保留原 mean、GEMM、FA4、缓存、KV、SPS/confidence、采样及 graph-off 配置。没有写替代 CUTLASS 的 GEMM，也没有量化实验。

## 无 profiler 的固定 A/B

工作负载仅 `p2048 × c1 × output128`，完整 prefill。先用 baseline 两个 64-request 窗口通过噪声检查，再测独立 AB；每个 arm 都完成 39 个串行/4/8 并发 correctness 请求及预设热身。首个 AB 已失败，因此**没有运行 BA 或扩大矩阵，也没有重跑/丢弃任何稳态样本**。

| 指标 | baseline，64 requests | fused，64 requests | fused 相对变化 |
| --- | ---: | ---: | ---: |
| TTFT mean | 255.717ms | 255.769ms | +0.020% |
| TTFT p95 | 256.709ms | 266.984ms | **+4.002%（退化）** |
| TPOT mean | 18.9681ms | 18.9107ms | −0.302% |
| TPOT p95 | 18.9987ms | 18.9948ms | −0.020% |
| E2E mean | 2664.853ms | 2657.606ms | **−0.272%（不足 1%）** |
| E2E p95 | 2668.882ms | 2683.597ms | **+0.551%（退化）** |
| 稳态设备级显存采样 | 57005MiB | 57005MiB | 未观察差异，非 allocator peak |

五项 acceptance 逐请求与冻结 oracle 完全一致；64-request 窗口的汇总为：correct_drafts=5760、proposed_drafts=17472、verify_count=2496、draft_accept_rate=0.32967032967032966、completion_tokens_per_verify=3.282051282051282。这里只做 full-prefill kernel A/B，不把 shadow/paired/partial recovery 缓存结果混入。

全部保存并 live-check 的请求：192（control）、136（AB baseline）、128（AB fused），合计 **456**。三个 arm 的 correctness 各 39 请求，共117；热身分别24/32/24，均依据预注册最后三组规则 admission。所有 steady 共256请求（control128+AB128）。此前 h48/h49 两条短 nsys 路径各有28个保存的 checked请求。GPU parity 25组、CPU协议与归因测试5项均通过。

baseline/control 的均值 spread：TPOT0.016676%、E2E0.006304%；p95 spread：TPOT0.016171%、E2E0.061548%，均低于预设噪声门槛。候选 p95 容忍仍是零，未因 control admission 放宽。

状态：`safety_stopped`，首个停止原因为 TTFT p95。`comparisons.jsonl` 在 guard 之前保存完整 ratios，并独立记录 E2E≥1%为false。共享 p95 检查函数的报错文字沿用 `full/paired` 名称；实际服务 arm 明确是 **baseline/fused**，两侧都 full-prefill，不是 paired-cache A/B。

## 为什么不把 kernel 数减少写成成功

h48/h49 各80个实际decode的 launch-correlated nsys 归因确认：residual norm kernel 数990→270，总kernel数1636→916；norm累计GPU时间1617.43→405.83µs/轮。但全部kernel累计GPU时间59985.99→59943.02µs/轮，几乎不变；GEMM在两条trace中的时间变化抵消大部分局部下降。

无 profiler A/B 也只观察到0.272%的mean E2E改善。**不能以局部GPU时间、kernel数量或其他项目的2.1%报告替代本项目端到端验收。**

只读遥测分析显示AB两侧steady显存均57005MiB，平均SM时钟约2417.94/2417.97MHz，显存时钟均12481MHz；采样clock-event-reasons均0。未观察到稳态采样层面的硬件限频证据，但这不等于排除了短暂扰动。TTFT尾部抖动的具体来源、本候选真实critical-path收益小的因果解释仍**未证明**，不归咎驱动或硬件，不删除抖动请求来重新算“达标”。

## 下一方案选择

**不继续打磨该两阶段 norm candidate、不写更多 GEMM。接下来研究优先项是 graph catalog / 纯设备 metadata prep 的安全可捕获边界，先做覆盖率与成本分析。** 这还是待验证方案，不宣称一定有收益。

- 保留 exact geometry guard 与 mismatch eager fallback。先统计实际 `(request count, ordered verify_lens, live total)` 分布，再决定有限catalog是否有足够覆盖率；同时核对页表归属、padding、hidden-mode、split和persistent-buffer lifetime。相同总token数不能代表同一个geometry。
- 只考虑由纯GPU kernel填充persistent buffer的metadata子图，host FlashInfer plan不得捕入。当前baseline是FA4、graph-off，不能直接移植FA2/EAGLE的sync-free-plan PR。
- CUTLASS epilogue暂留独立研究：保留library mainloop；当前安全baseline trace的主GEMM是已有batch-invariant `matmul_kernel_persistent`，不是已经证明可接EVT的CUTLASS实现；官方新Operator教程又要求SM100f，而本机SM120。先证明兼容性与收益预算，不直接替换算术配置。
- KV写回与greedy采样在已分析的c1 trace中分别约61/44µs/轮，目前不足以支持优先新增kernel。概率采样是另一个workload，FlashInfer的sorting-free top-k/top-p报告不能套到greedy。

具体文章、社区PR及已阅读diff见同目录《研究依据-20261008.md》。

## 回退与可复现性

- 服务HEAD仍`0577109f4e659b8ef2ef3f979f1fd7f079b25821`，tag为`dspark-stable-fa4-fixed64-matrix128-passed-20261007`；tracked工作树干净，GPU无剩余实验服务。
- main未修改；当前既有未跟踪文件仍8357项。冻结的执行源码、包、model metadata、SPS表、oracle在h50结束时integrity通过。未删除任何baseline文件。
- 独立代码分支`exp/residual-rmsnorm-fusion`。实施`b7570bd`；归因`a3897a2`；无profiler协议`0c215c0`。后续分析/结果文档是事后提交，不改已测kernel。
- 原始记录：`/root/autodl-fs/dspark-residual-norm-baseline-profile-20261008-h48`、`...candidate-profile-20261008-h49`、`...unprofiled-ab-20261008-h50`。
- h50 manifest固定oracle、工具SHA、experiment head、GPU UUID、包及源码。没有新oracle；没有把baseline兼容旧BF16路径当作新的验收条件。
- 全部失败记录保留，候选仅供归档/进一步研究；不能作为已经成功的1%优化成果。
