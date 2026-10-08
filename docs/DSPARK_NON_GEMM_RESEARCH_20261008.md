# DSpark 非 GEMM 融合研究依据（2026-10-08）

## 选择：先验证 residual RMSNorm 点运算融合

不是替代 CUTLASS 的 GEMM。保留当前已资格化 baseline 的 GEMM、FA4、mean 归约、KV、SPS/confidence、采样、缓存和 graph 配置。本轮只写两个 Triton **点运算** kernel；不是 Triton GEMM。

独立分支：`exp/residual-rmsnorm-fusion`。服务仓库仍位于 `0577109` / `dspark-stable-fa4-fixed64-matrix128-passed-20261007`，实验以独立进程 hook 加载，进程退出即撤销。没有改 main，没有删除既有未跟踪文件。

## nsys 证据及其边界

现有 h43 32-request trace 显示：每轮实际 decode 约 1636 个 kernel；KV 写回约 61µs，greedy argmax 合计约 44µs；残差 norm 相关 cast/add/square/mean/rsqrt/mul 链重复约 90 次。只看名字尚不能完成归因，因此新增 residual norm NVTX 范围，并按 PID + runtime launch correlation 归属 GPU kernel。

h48 原路径、h49 融合路径各采集两个 128-token 请求，均先有 24 个 excluded settle 请求。每侧覆盖 **80 个实际 decode**，不是仅按返回的 39 次 verify 统计；overlap 包括 lookahead。完整输出及五项 acceptance 逐请求对照冻结 oracle。

| trace 观察 / 每轮实际 decode | h48 原路径 | h49 融合路径 |
| --- | ---: | ---: |
| residual norm 调用数 | 90 | 90 |
| residual norm kernel 数 | 990 | 270 |
| 总 kernel 数 | 1636 | 916 |
| residual norm GPU 时间累计 | 1617.43µs | 405.83µs |
| 全部 kernel GPU 时间累计 | 59985.99µs | 59943.02µs |

候选确实把 norm 的 11 个 kernel 改为 3 个；其中 mean 仍是原 kernel。**总 profiled GPU 时间几乎没变**，不能由 norm 局部减少直接宣布端到端加速。GEMM 在两条 trace 中占用变化，也说明 profiler、时钟、调度等扰动不能忽略。trace 只能用于归因，性能必须另用无 profiler 的固定 A/B。

适用范围：BF16、连续二维张量、H5120、M1..64、有 residual、eager。其他布局、dtype、形状、norm 选项或 graph capture 均保持原路径。保留 FP32 add/square/variance/multiply 与最终 BF16 舍入；禁止编译器 FP fusion，未换归约拓扑。

25 组 GPU 抽样逐位测试（rows=1/4/8/32/64 × scale=0/1e-20/1e-3/1/1000）均通过，output/residual bit diff=0。另有 5 项 CPU 测试覆盖噪声门槛、p95 零容忍、runtime opt-in、跨 PID correlation 与缺失步骤拒绝。**不将抽样 parity 表述为所有输入的数学证明。**

## 文章与社区 PR 给出的可迁移思路

1. [NVIDIA：Kernel Fusion（2026-07-10）](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)

   融合 kernel 本体可减少中间访存与启动；CUDA Graph 只合并 host dispatch，并不自动消除中间 tensor。对本项目的启发是先找高频点运算链，而不是重写 GEMM。文章示例是其他硬件与工作负载，不能移植其倍数收益。

2. [LMSYS：Ling-3.0-flash on Blackwell（2026-08-21）](https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell/)

   要区分 host 空隙与 GPU 小 kernel 成本，并校准 profiled/unprofiled。融合必须保持原 fast-math 和中间舍入边界。文章还讨论纯设备 persistent-buffer metadata 的小图，以及避免 FlashInfer plan 的阻塞 D2H；不能把 host plan 捕进第二张图。其 Ling 混合 MoE、TP4、SM100 相关收益不等于本项目 Qwen3 dense、TP1、SM120 的收益。

3. [CUTLASS：Custom epilogue fusions](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)

   官方支持往已有 GEMM Operator 添加 activation、辅助输出和广播等 epilogue，不必改 mainloop。**该 tutorial 显式要求 sm_100f**，不能默认它在本机 SM120 可用。今后若 profile 证明值得做，应在兼容的既有 CUTLASS 实现上研究 epilogue，并保留已有 GEMM 输出的 BF16 舍入边界。

4. [FlashInfer：Sorting-Free Sampling（2025-03-10）](https://flashinfer.ai/2025/03/10/sampling.html)

   思路适用于带 top-k/top-p/min-p 的概率采样，不能把 sorting-free 优势套到 greedy argmax。当前 greedy trace 采样占比小，因此本轮不新增采样 kernel。若未来概率采样 workload 成为主要成本，应单独测试 RNG、过滤语义、分布、acceptance 与 p95。

5. [SGLang PR #37300：ragged 请求/token 容量解耦](https://github.com/sgl-project/sglang/pull/37300)（2026-09-02 合并）

   已阅读实际 diff：提前保存 request-slot max_bs，token buffer 按实际 capture token grid 的上界分配。它解决容量管理，不证明不同 row geometry 可以安全 replay。本仓库 exact guard 仍不变。

6. [SGLang PR #28854：sync-free fast_prefill_plan](https://github.com/sgl-project/sglang/pull/28854)

   已读取公开 diff：从 host-known qo/kv metadata 调用 cached plan，避免 per-replay GPU→CPU；实现严格限定 FA2、EAGLE draft-extend、已有 cached_module、完整单 wrapper，并要求 host 参数显式传入；GPU 测试包含反转 KV 索引的负对照。本项目当前 FA4 graph-off baseline 不满足该实验的直接接入条件，故不盲目移植。

7. [SGLang PR #37329：graph/spec 输出处理](https://github.com/sgl-project/sglang/pull/37329)（2026-09-03 合并）

   已读取最终公开 diff：可研究单 aux hidden state 免 cat、target hidden projection 在 sequence-parallel gather 前执行以及模型侧 graph-layer hook。PR 中最初的 customized-output snapshot 已另行撤回，不能照初稿宣称保留。TP1 无跨 rank gather，该部分收益不能直接推断。

8. [vLLM PR #40413：batch-invariant residual RMSNorm 融合](https://github.com/vllm-project/vllm/pull/40413) 与 [PR #48391：固定 RMSNorm reduction block](https://github.com/vllm-project/vllm/pull/48391)

   已阅读公开讨论与实际 diff。前者在自己的 FP8 Llama latency 测试报告约 2.1% 改善，直接支持“batch-invariant 模式下残差 norm native 链值得 profile”的方向，但不是本项目收益。后者修正 token 数跨 256 时改变 block size、从而改变浮点求和顺序的问题，并增加跨阈值测试。启发是固定归约拓扑与融合都要有独立 correctness 证据；“新 kernel 自身 batch-invariant”也不代表它复现本项目已冻结的原算术结果。本候选因而保留原 mean，而不是直接切到另一套 fused RMSNorm reduction。

## 后续方向与止损

本轮先完成一个隔离候选的无 profiler A/B，不叠加其他方案。h50 的预注册协议为：p2048/c1/output128，baseline 两个 64-request 窗口确认噪声，然后 AB/BA；每个服务 arm 都有 39 个串行/1/4/8 并发 correctness 请求、24..64 warmup，64 个 steady 请求。两个方向均要求 mean E2E 至少降低 1%，TTFT/TPOT/E2E p95 不能退化；不重跑失败 control，不挑选 steady 样本，总预算 40 分钟。

Graph 若以后继续，应先记录常见 **完整 row-lens pattern** 与资源/页表约束，再预捕获有限 catalog；不是用相同 token total 代替相同几何。当前 runner 除 exact lens/row match 外还要求 live total==tier、单请求等条件。多请求 catalog/page-table 重排需单独证明正确性、buffer lifetime、padding、hidden projection 和 split 策略，任何 mismatch eager fallback。另可独立分析纯 metadata kernel 链，但不要把 host plan 或动态分配捕入小图。

此文中的理论路线与 trace 局部数字都不是 ≥1% E2E 成功证明。最终结论以 h50 的保存结果及门槛为准。
