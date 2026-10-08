# 固定 verify-all metadata 快路径：结果与止损

结论：**没有证明达到 1% 端到端收益，不升为默认方案。**

固定工作负载：Qwen3-14B + DSpark block7，prompt2048、并发1、输出128。
两臂均为完整 prefill，paired cache/graph/动态 draft length 关闭，BF16、
FA4 fixed64、Torch batch-invariant GEMM、greedy PyTorch 保持原样。
候选显式省略 SPS 表，复用已有 uniform compact layout，仅跳过不再使用的
host budget planner；confidence head/relay、KV、采样、commit/readiness 不变。
这是固定验证策略实验，**不是**“自适应满预算天然等价于完整 verify”的证明。

## 实测首轮 AB

每臂64个正式请求，相同的39个正确性预处理请求、24个已通过稳定性准入的
预热请求和1个 priming 请求；服务内置启动预热不计入正式窗口。
以下变化是候选相对控制的时间变化，负值表示更快。

| 指标 | 旧策略 A | 固定策略 B | 变化 |
|---|---:|---:|---:|
| Mean E2E | 2666.545 ms | 2652.815 ms | **-0.515%** |
| Mean TPOT | 18.984908 ms | 18.873908 ms | -0.585% |
| Mean TTFT | 255.283 ms | 255.648 ms | +0.143% |
| E2E p95 | 2677.499 ms | 2655.990 ms | -0.803% |
| TPOT p95 | 19.045679 ms | 18.911576 ms | -0.704% |
| TTFT p95 | 258.988 ms | 257.378 ms | -0.622% |
| Acceptance（timing variant0） | 32.967033% | 32.967033% | 相同 |
| 采样 device 显存峰值 | 57005 MiB | 57005 MiB | 相同 |

显存是200ms周期 NVML 设备级采样峰值，不是 PyTorch allocator 峰值。
TPOT是客户端 `(last_token_time-first_token_time)/127`，没有把 speculative
burst 人工拆成虚假的逐 token 间隔。E2E 包含客户端完整 streaming 响应结束。

首轮节省13.731 ms；1%门槛需要至少26.665 ms。三个 p95 guard 均通过，
没有 token/acceptance 错误或非法访问，但均值收益不达标。
事后按相邻8请求分块的条件 bootstrap，时间下降区间约0.438%–0.608%；
这只是窗口内描述，不覆盖跨窗口漂移，不是部署置信度，也不能替代换序复验。

预先要求 AB、BA **各自**至少1%且 p95 零退化。AB 已不满足，继续 BA 无法
让这一合同通过，因此停止剩余计时。**BA 未完成，没有可重复收益证明。**
B2 已完成39个正确性请求，但没有正式计时；A2 未运行。所有已完成请求保留，
没有删除慢请求、重试生成或更换参考。后续脚本增加了同一门槛的自动早停，
并非改变已执行实验的验收标准。

## 正确性、控制噪声与范围

- 新实例 baseline 已对照旧冻结 h32 oracle 验证 p2048 的串行/4/8并发。
- 候选 B1、B2 各39个请求覆盖全部13个 variant 的串行和两轮并发。
- h47及继承的控制记录合计487个完成请求，256个正式控制/AB计时请求。
  逐 token 和五项 acceptance 字段审计失败数为0；cached_tokens与retractions均为0。
- 初始 full/full 两窗口各64请求：TPOT和E2E均值差约0.006%，p95差均低于0.03%，
  通过预定控制准入。没有反复重跑失败控制直到通过。
- 只证明这些输入的 correctness；没有运行新的全prompt性能矩阵，未验证 paired
  reuse、partial recovery、长于128输出、其它 GPU 或生产流量收益。
- 由于启动修复，A1与B1之间有额外时间间隔；没有完成 A2 控制漂移复验。
  即使观察到0.515%改善，也不将它称为已复现的因果加速。

## 修改、启动问题与可回退性

服务 checkout 始终为 `0577109` / `dspark-stable-fa4-fixed64-matrix128-passed-20261007`，
tracked源码、旧oracle和既有未跟踪文件未改写；main未修改，没有push。
候选为默认关闭的进程内启动适配器，退出服务即撤销。它是实验适配器，不是已
验收的生产接入。提交在独立 `exp/fixed-verify-all-ab` 分支。

保留了全部问题记录：

1. h44：发现A/B正确性预处理不对称，控制采集中止；没有候选数据。修正为每个arm
   相同预处理，原两小时截止时间不顺延。
2. h45：控制准入与A1完成，但 forward hook 安装晚于 planner 构造，候选 attestation
   失败。没有候选正确性/计时数据。
3. h46：同一启动顺序下 graph runner 字段尚未建立，启动拒绝；没有候选数据。
4. h47：修复启动适配，仅只读继承h45已通过控制，不重跑、不挑窗口。B1首次完成
   候选计时；AB收益不足1%，16:15左右止损并关闭全部自有GPU进程。

上游也确认 forward hooks 后置安装：[SGLang graph/hook setup源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py)。
启动期只查找唯一 planner、检查 cache 尚未创建，先验证实际 prefill/decode graph
backend均为DISABLED；不在 decode热路径扫描对象，不放宽 exact geometry guard。

测试：新增候选/验收测试15项，加既有稳定性7项和fixed-policy4项，共26项通过；
原有ragged verify10项、backend capability2项在新实例通过。

硬件：RTX PRO6000 Blackwell Server Edition / SM120，97887MiB，驱动595.71.05；
GPU UUID变更为GPU-7361f058-4f4f-325a-0075-2b03e0838996。UUID迁移记录独立保存，
旧资料未改写，新实例正确性实际重验。oracle SHA256保持
`f0ec55fe326b7c8d253879a08a9d11c9c6d795b1aa4f41926c9da3f05626f5a1`。

## 理论复盘与决定

这次观察支持“移除预算/布局工作有小幅改善线索”，但不支持“它足以带来1%”。
历史 nsys 中 schedule_layout 的约3.6ms host wall包含GPU等待、重叠与采集扰动，
不能全部视作可删除成本。本次真实E2E改善量远小于按该wall时间直接相加的估计。
不能据此断言 GEMM 接近最优，或证明 TPOT 尖峰的根因已经解决。

按事先选择的投入边界，**封存这一 narrow metadata 候选，不继续扩大矩阵或叠加
confidence/head/relay、GEMM、graph、量化改动来凑1%**。下一方向需独立选题和合同。
paired cache的旧p95失败依旧有效，本次不是paired cache收益证明。

原理参考，不迁移他人的收益数字：[Modal host overhead](https://modal.com/blog/host-overhead-inference-efficiency)、
[SGLang Spec V2文章](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)。
