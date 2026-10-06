# DSpark 优化实现与验证记录

本文档记录本仓库中 DSpark 相关的每个可回退提交、它解决的问题、已观察到的效果、依赖关系和下一步计划。文档对应的代码分支是 `exp/reuse-compact-index`；`main` 保留基线，`exp/manual-fused-kv-gemm` 保留手写 GEMM 实验。

## 状态约定

- **主线/安全**：已进入当前实验主线，默认行为保持安全；没有把未经验证的加速强行打开。
- **正确性修复**：主要解决输出、acceptance、缓存生命周期或动态长度问题，不宣称端到端加速。
- **可选实验**：通过环境变量或独立分支启用；目前没有足够的稳定态服务 A/B 数据，不把它写成收益。
- **保护性回退**：检测到不支持或几何不匹配时回到 eager/完整 prefill，优先保证正确性。

## 总体依赖关系

```text
0c1f1eb baseline fused KV write
  ├─ 11489a1 greedy verify finalize metadata
  ├─ 440f2dd acceptance determinism guards
  ├─ e239058 compact row-index reuse
  └─ 4a2efd4 paired target/draft sidecar
       └─ a6f025a readiness + paired matching
            ├─ 5384a74 shadow mode read-only
            ├─ 555af1f fingerprint invalidation
            └─ 7de69fe partial paired-prefix recovery

2a41311 compact/scan CUDA path
  └─ 76a675e adaptive draft verify length
       ├─ 35abf84 planner import fix
       ├─ 531501e length bucketing
       └─ ff74243 unsupported-graph fallback
            └─ a1e97c0 FlashInfer ragged verify graph (opt-in)
                 ├─ 4b26a46 live ragged-length compact IDs
                 └─ ee2074d exact capture/runtime geometry guard

独立实验：
  6c7fb13 manual fused KV GEMM branch
  1c6e7bb CUTLASS epilogue visitor probe
  4eb07f4 FlashInfer logits top-k-first sampler
```

## 逐提交说明

### 0c1f1eb — `baseline DSpark fused KV write path`

- **做了什么**：加入 DSpark Qwen3 dense 路径、fused KV write Python/CUDA 调用骨架以及模型侧接入。
- **解决的问题**：建立后续 speculative verify、draft KV 写入和 compact window 优化的共同基线。
- **效果**：这是当前所有实验的功能基线，不把它与未优化 SGLang 的整体差异混为单个 kernel 收益。
- **后续**：用 nsys 将 fused KV write、verify、sampling 分开测量，再决定是否值得改 epilogue 或内存布局。

### 6c7fb13 — `experiment manual fused DSpark KV GEMM write`（独立实验分支）

- **依赖**：0c1f1eb。
- **做了什么**：在 `exp/manual-fused-kv-gemm` 上试验不套普通 Triton GEMM 的手写 KV GEMM/write 路径。
- **解决的问题**：验证手写 kernel 是否能把 GEMM、投影结果和 KV 写入合并。
- **效果**：目前没有稳定态端到端收益证据；该分支独立保留，未覆盖当前主线，也没有把普通 Triton GEMM 当成最终方案。
- **后续**：只有在 nsys 证明 KV write/访存占比足够高，并完成多并发 A/B 后，才继续做 tile、向量化和 epilogue 调优。

### e239058 — `reuse compact row index across DSpark verify window`

- **依赖**：0c1f1eb。
- **做了什么**：缓存 DSpark verify window 的 compact row index，减少同一 verify 窗口内重复构造和搬运索引。
- **解决的问题**：compact/scan 路径的小型索引开销和重复 metadata 工作。
- **效果**：属于低风险的索引复用优化；尚未形成独立稳定态服务端收益数字，需和动态 draft length 一起用 nsys 复核。
- **后续**：与 2a41311 的 CUDA compact/scan 合并评估，确认索引复用没有让动态长度语义变陈旧。

### 1c6e7bb — `wip: probe CUTLASS epilogue KV visitor for DSpark`（可选实验）

- **依赖**：0c1f1eb。
- **做了什么**：加入 CUTLASS epilogue visitor 的探针 CUDA 文件，研究在保留 GEMM 主循环的情况下融合 KV 写入。
- **解决的问题**：验证“保留接近 SOTA 的 CUTLASS GEMM，只改 epilogue/写回”是否比重写 GEMM 更现实。
- **效果**：是探针，不是默认路径；当前没有证明端到端收益，也没有把它声明为已完成优化。
- **后续**：先通过 nsys/ncu（若环境开放）确认 epilogue 和 cache write 的占比，再决定是否补齐编译、注册和 A/B。

### 4eb07f4 — `opt-in FlashInfer logits top-k-first sampling`（可选实验）

- **依赖**：DSpark sampling/verify 基线。
- **做了什么**：增加 FlashInfer logits top-k-first 的 opt-in sampling 路径。
- **解决的问题**：探索 lm-head 后处理、top-k/argmax 和采样之间的中间 tensor/launch 开销。
- **效果**：未证明稳定态端到端收益；默认不启用，不能和 fused KV 或 prefix cache 的效果相加估计。
- **后续**：只有在采样占比足够高时，才继续做 joint softmax/top-p/采样 kernel；同时检查 greedy、top-k、top-p 语义一致性。

### 11489a1 — `fuse greedy verify finalize metadata`

- **依赖**：0c1f1eb。
- **做了什么**：把 greedy verify 的 acceptance finalize、位置/长度 metadata 更新集中到 DSpark 专用路径。
- **解决的问题**：减少 verify 完成阶段的 Python/小 kernel 往返，并统一后续 cache/acceptance 所需状态。
- **效果**：为 acceptance determinism 和 paired readiness 提供稳定状态边界；没有单独宣称大幅吞吐提升。
- **后续**：配合 440f2dd 检查 finalize 的 stream 顺序，避免“metadata 已更新但 KV 写入尚未完成”。

### 440f2dd — `make acceptance deterministic with prefix cache guards`

- **依赖**：11489a1。
- **做了什么**：为 prefix cache、CUDA graph、DSpark scheduler 增加 correctness-first 的 acceptance 保护和确定性开关。
- **解决的问题**：缓存命中顺序、动态 graph 和并发条件下 acceptance/output 不稳定。
- **效果**：在 `--enable-deterministic-inference` 且 ragged graph 关闭的安全配置下，串行和并发 greedy 输出稳定，观测 acceptance mean 约为 `2.2857142857142856`。此前未经保护的 graph 场景出现过串行/并发 hash 不同，因此不能把 graph replay 直接当作正确路径。
- **后续**：先恢复精确 graph capture 几何，再做性能 A/B；不要用放宽 guard 换取不确定的输出。

### 4a2efd4 — `add paired target draft prefix sidecar`

- **依赖**：440f2dd 的安全缓存边界。
- **做了什么**：增加 DSpark 专用 sidecar，把 target radix node 的 token location 与 draft KV readiness metadata 绑定。
- **解决的问题**：只命中 target radix cache 时 draft KV 缺失，不能安全复用 target-only prefix。
- **效果**：建立 paired prefix cache 的数据结构和生命周期入口，不改变通用 target KV 语义；默认关闭，sidecar 无效时回退完整 prefill。
- **后续**：完成 readiness、fingerprint、eviction/split、budget 和真实命中率测试。

### a6f025a — `enable paired matching and draft readiness`

- **依赖**：4a2efd4。
- **做了什么**：在 scheduler/worker 中接入两阶段 paired match，并在 draft KV 写入完成后记录 ready 状态。
- **解决的问题**：target/draft 前缀错配，以及 scheduler 只按 target cache 截断长度。
- **效果**：最终 prefix 长度同时受 target KV 和 draft KV 约束；target-only 命中但 sidecar 不完整时安全回退。暂未宣称 TTFT/TPOT 提升。
- **后续**：用重复 prompt、不同 suffix、并发、abort/retry 和 eviction 做 token-by-token 一致性矩阵。

### 5384a74 — `keep shadow mode read-only`

- **依赖**：a6f025a。
- **做了什么**：将 shadow mode 限定为观测 paired hit、ready 长度和 fallback 统计，不改变调度结果。
- **解决的问题**：在尚未证明 sidecar 完整性前，避免观测逻辑污染线上 cache 行为。
- **效果**：可以先测真实 paired hit rate；默认 correctness 不受 shadow metadata 影响。
- **后续**：当统计证明 paired hit 足够高，再单独开启 opt-in paired matching。

### 555af1f — `invalidate paired metadata on fingerprint change`

- **依赖**：4a2efd4、a6f025a。
- **做了什么**：把 draft 权重/配置指纹写入 sidecar，并在指纹变化时让 paired metadata 失效。
- **解决的问题**：draft 权重 reload、KV dtype、TP/PP、RoPE 或投影配置变化后复用旧 draft KV。
- **效果**：避免 stale draft KV 造成错误 acceptance；增加了相应 cache 单测。
- **后续**：补齐真实 reload、cache reset 和 target eviction 的联动测试。

### 7de69fe — `add partial paired-prefix recovery plan`

- **依赖**：a6f025a、555af1f。
- **做了什么**：允许 sidecar 只恢复最长连续 ready 区间，而不是把整棵 target prefix 当成全命中。
- **解决的问题**：部分 draft KV 缺失时，target-only radix 命中长度与 paired 可用长度不同。
- **效果**：把不完整 sidecar 转成可控的 partial recovery；缺失区间仍然走安全 prefill。
- **后续**：增加 sidecar 25% budget 的淘汰和恢复比例测试，并记录 TTFT 是否真的下降。

### 2a41311 — `add optional compact scan CUDA kernel`

- **依赖**：e239058 的 compact row-index 语义。
- **做了什么**：加入可选 `dspark_compact_scan.cu` 及 Python 封装，把 compact/scan 的小型索引工作移到 CUDA。
- **解决的问题**：verify window 中 compact、prefix sum 和索引写回的 launch/同步开销。
- **效果**：功能上是 opt-in；没有把单个 microbenchmark 推断成端到端收益。
- **后续**：用 nsys 确认 compact/scan 的总时间占比，验证动态长度和 graph capture 下的 fallback。

### 76a675e — `add adaptive draft verify length`

- **依赖**：2a41311。
- **做了什么**：加入 DSpark planner，根据运行时反馈选择 draft verify 长度。
- **解决的问题**：固定 block size 在不同 acceptance、prompt 长度和并发下不合适。
- **效果**：为动态 draft length 提供策略层；默认仍受安全开关和后端能力限制。
- **后续**：将 acceptance、verify cost 和 queue pressure 纳入 cost table，避免 planner 只追求更长 draft。

### 35abf84 — `fix dynamic planner type import`

- **依赖**：76a675e。
- **做了什么**：修复 planner 接入 worker 时的类型导入问题。
- **解决的问题**：动态 draft length 路径启动/导入失败。
- **效果**：纯工程修复，不产生独立性能收益。
- **后续**：把 planner 的类型和配置检查加入单测，防止回归。

### ff74243 — `gracefully fallback when ragged graph unsupported`

- **依赖**：76a675e、CUDA graph setup。
- **做了什么**：后端不支持 ragged verify graph 时明确回退 eager 路径。
- **解决的问题**：把不支持的 graph capability 当成可用，导致启动失败或错误 replay。
- **效果**：不支持时服务仍能安全运行；这是保护性回退，不是加速。
- **后续**：把 capability、capture geometry 和 runtime geometry 分成独立检查。

### 531501e — `bucket adaptive verify lengths`

- **依赖**：76a675e、ff74243。
- **做了什么**：把动态 verify length 映射到有限 bucket，降低 graph shape 组合数量。
- **解决的问题**：每个长度都 capture 会造成 graph 数量、显存和管理开销爆炸。
- **效果**：减少 shape 组合，但 bucket 仍可能与真实 ragged row geometry 不同，因此不能单独保证 replay 安全。
- **后续**：与 ee2074d 的 exact geometry guard 配套，只在精确匹配时 replay。

### a1e97c0 — `add guarded FlashInfer ragged verify graph`

- **依赖**：ff74243、531501e。
- **做了什么**：接入 opt-in FlashInfer ragged target-verify graph、capability 标记和 graph admission 逻辑。
- **解决的问题**：希望减少 target verify 的重复 launch，同时处理后端不支持情况。
- **效果**：建立了 graph 路径，但在动态长度/并发下曾观察到不一致风险；因此默认关闭，不能宣称端到端收益。
- **后续**：精确 capture/runtime geometry、row count、host mirror 和 stream 顺序全部一致后，才重新做稳定态 A/B。

### 4b26a46 — `keep compact verify ids aligned with live ragged length`

- **依赖**：a1e97c0。
- **做了什么**：让 compact verify ids 使用 live total/padded total，而不是 stale graph bucket 上界。
- **解决的问题**：动态 ragged 长度下 compact ids 与实际行数不一致，可能读写越界或污染后续请求。
- **效果**：修复动态长度 metadata 对齐；对应 ragged verify 单测通过，安全 eager 路径没有再复现此前的非法访问。
- **后续**：继续验证 graph capture 和 eager 两种模式的相同 token 输出。

### ee2074d — `require exact geometry for ragged verify graphs`

- **依赖**：a1e97c0、4b26a46。
- **做了什么**：要求 capture 时保存的 verify lens、host mirror、row count 和 runtime geometry 完全一致，否则不 replay，回退 eager。
- **解决的问题**：仅按 bucket/capability 判断 graph 可用会在实际行数变化时复用错误 graph。
- **效果**：把不确定的 graph replay 变成保守的 exact-match admission；强制开启 graph 时不满足几何条件也会安全回退。当前默认 graph 关闭，单测 `test_ragged_verify.py`（10 项）和 backend capability 测试（2 项）通过。
- **后续**：不要放宽 exact guard；若要取得 graph 收益，应做按几何模式的 graph catalog 或 page-table 重排，再进行 serial/4/8 并发 A/B。

## 当前可复现实验结论

1. **确定性**：安全配置是 `--enable-deterministic-inference`、ragged graph 关闭、未经证明的 target-only prefix 不复用。该配置下串行/并发 greedy 输出稳定；早期未加 guard 的 graph replay 曾出现不同 hash，因此被视为已修复的 correctness 风险。
2. **Ragged graph**：当前贡献是能力探测、精确几何 guard 和安全回退，不是已证明的性能优化。强制开启但几何不匹配时仍应走 eager。
3. **Paired prefix cache**：已经具备 sidecar、ready、fingerprint、partial recovery 的实现骨架；需要真实服务端命中率和 TTFT A/B 才能判断收益。
4. **Compact/scan 与动态 draft length**：已具备可选 kernel 和 planner，但要用 nsys 证明它们在总时延中占比足够高；不能从单个 kernel 时间直接推导端到端提升。
5. **手写 GEMM/CUTLASS epilogue/sampling**：均保留为独立实验或探针。CUTLASS GEMM 主循环接近最优时，优先优化 epilogue、KV 写回、索引和融合边界，而不是写普通 Triton GEMM。

已运行的回归命令：

```bash
PYTHONPATH=python python test/registered/spec/dspark/test_ragged_verify.py
# 10 tests OK

PYTHONPATH=python python test/registered/spec/dspark/test_ragged_verify_backend_capability.py
# 2 tests OK
```

## 推荐的 GitHub 回滚/查看方式

当前主线提交链保留在 `exp/reuse-compact-index`；基线是 `main` 的 `0c1f1eb`。建议给以下里程碑打 tag，便于逐个 A/B 和回退：

- `dspark-baseline-kv-write` → `0c1f1eb`
- `dspark-deterministic-acceptance` → `440f2dd`
- `dspark-paired-prefix-sidecar` → `4a2efd4`
- `dspark-paired-prefix-ready-match` → `a6f025a`
- `dspark-paired-prefix-partial-recovery` → `7de69fe`
- `dspark-compact-scan` → `2a41311`
- `dspark-adaptive-draft-length` → `76a675e`
- `dspark-ragged-verify-graph` → `a1e97c0`
- `dspark-ragged-verify-exact-geometry` → `ee2074d`
- `dspark-manual-fused-kv-gemm` → `6c7fb13`

## 下一步优化优先级

1. **先做 paired cache 的真实收益闭环**：shadow hit rate、partial recovery、eviction/fingerprint、4/8 并发和 128/512/2K/8K prompt 的 TTFT/TPOT/p95。
2. **再做 ragged graph 的可用形状覆盖**：按常见 verify geometry 建 graph catalog，或者重排 page-table 使 runtime geometry 命中已 capture 的形状；每次只放开一个形状并做 token-by-token 对照。
3. **继续 compact/scan 融合**：只有当 nsys 证明 compact/scan 占比高且同步开销明显，才扩展 CUDA kernel；动态长度必须保留 live total 语义。
4. **重新评估采样融合**：在 logits 后处理成为瓶颈时，比较 FlashInfer、手写 joint sampling 和框架调度开销，先验证语义再看收益。
5. **单独建立 Blackwell/FP4 量化分支**：量化收益不能与上述 cache/graph 优化混算；需验证 FP4 计算路径、累加精度、KV dtype 和实际硬件支持。
6. **所有实验按提交回退**：任何输出不一致、acceptance 随命中顺序变化、p95 变差、sidecar stale 或 graph 越界，立即回到上一里程碑 tag。
