# DSpark 固定 cache A/B harness：实现与实跑记录

## 交付状态

harness 已实现；完整 cache A/B **未完成，收益未证明**。
原因为完整 prefill 对照组出现串行/并发逐 token 差异。没有通过放宽 gate
继续扫表，也没有修改任何推理、kernel、量化或 graph geometry 代码。

代码保存在远端仓库的 `exp/paired-cache-ab-harness` 独立分支：

- `bbe21ba`：固定 streaming cache A/B harness。
- `afc1a40`：独立 deterministic full-prefill 诊断 profile。
- `5d8b2b5`：full-prefill cache 被禁用时，sidecar 统计标记 unavailable，而非伪造为零。

仅新增 `benchmark/dspark/` 的脚本、CPU 单测和协议文档。未 push，未修改 main。
原有 8357 个未跟踪文件全部保留。服务均已停止；GPU 回到 0 MiB / 0% 利用率。

## 固定协议

- Prompt 精确为 128 / 512 / 2048 / 8192 个 token ID，与 1 / 4 / 8 并发交叉，共 12 个 cell。
- full-prefill、paired cache、显式 opt-in 的 unsafe target-only-shadow 三种模式。
- 默认所有 CUDA graph 关闭，dynamic draft / CUDA compact-scan 关闭；不放宽 exact geometry。
- 输出长度固定，greedy + PyTorch sampler + `sampling_seed` + server seed。
- 每个 variant 先清缓存建立 cold reference；每个 repeat 再清缓存并 prime。
- 逐 token 输出与 acceptance counts/histograms 必须与 full reference 一致。
- 单独保存 cold、prime、steady 请求；只用 steady 请求统计性能。
- 真实流式 TTFT、客户端 amortized TPOT、TTFT/TPOT/E2E p50/p95/p99。
- 100 ms nvidia-smi 显存、利用率、温度采样；不是 PyTorch allocator 的瞬时 peak。
- 命中请求率、实际 cached-token rate、sidecar target/paired token hit rate、paired/target coverage、平均 paired hit length。
- partial/full recovery 的 plan 次数和 token 数、metadata evictions。
- 不存在独立暴露的实际 draft-worker fallback counter；不能把 recovery plan 当作实测 worker fallback。
- smoke：每 cell 8 个 steady 请求、1 repeat、32 output tokens。p95 仅描述。
- standard：每 cell 64 个 steady 请求、3 repeats、128 output tokens。满足样本阈值后 paired p95 比 full 变差即停。

## 实跑证据

### h2：普通 cache A/B，停在 full-prefill 对照

目录：`/root/autodl-tmp/results/dspark-cache-ab-smoke-20261007-h2/`

- 完成 1 个 cell：128 token、1 并发，8 个 steady 请求。
- 在 128 token、4 并发、variant 0 的 steady 请求停止。
- 第 3 个输出 token（0-based index 2）不同：串行 `34334`，并发 `16555`。
- 串行文本：`The document discusses a speculative decoding process ...`
- 并发文本：`The document describes a speculative decoding process ...`
- 两者 `cached_tokens=0`，尚未启动 paired / target-only-shadow 模式。
- correct/proposed drafts：串行 `22/70`，并发 `21/70`。
- 不能把这一差异归因于 paired cache，也不能由此确定某个 kernel 的根因。

### h4：deterministic + full-prefill + 全 graph-off 诊断

目录：`/root/autodl-tmp/results/dspark-cache-ab-deterministic-control-20261007-h4/`

- 完成 8 个 cell：128/512 × 1/4/8，以及 2048 × 1/4。每 cell 8 个 steady 请求。
- 已完成 cell 的每个响应均通过 token 和 acceptance gate。
- 在 2048 token、8 并发、variant 3 的 steady 请求停止。
- 第 7 个 token（0-based index 6）不同：串行 `16953`，并发 `1714`。
- 此运行仅为 full-prefill correctness 诊断，不是 cache 性能 A/B。
- 8192-token 三个 cell 未运行到；不能记成通过。
- FlashInfer 的 deterministic 配置禁用 radix cache，sidecar 缺失统计按 null/unavailable 处理。

### h1 / h3：联调兼容性记录

- h1 使用了错误的 native sampling 字段 `seed`，真实字段是 `sampling_seed`；已修复并加单测。
- h3 因 cache 被禁用后不存在 sidecar 而停止；已修复为 full-control 下 unavailable。
- 这两次未产生有效测量，不算模型正确性故障，也不纳入性能结果。

## 验证与下一步

- 新 harness CPU 单测：20/20 通过，本地和远端均执行。
- 原有 ragged verify：10/10；backend capability：2/2；paired prefix：6/6，均通过。
- CSV 和流式原始响应保留了已完成 cell 的指标；未运行的 cache mode 不填零、不猜测。
- **下一步应先定位 full-prefill 的 batch invariance 差异，并建立同时支持 deterministic 与 prefix reuse 的已验证配置。**
  当前服务参数校验仅允许部分 backend 在 deterministic 模式复用 radix；不能简单取消 FlashInfer guard。
- 在此之前不运行 standard cache A/B、不宣称 p95/acceptance/paired-cache 收益成立。

使用方式见同目录 `README.md`；所有失败及未完成结果保留，未删除已有 baseline 或日志。
