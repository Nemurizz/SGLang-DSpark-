# Original-arithmetic residual RMSNorm graph catalog

Independent default-off process-local experiment, based on e70cfd6 tooling.
The earlier fused residual-norm candidate is REJECTED and not installed/stacked.

## Theory and limits

Existing h43: 990 residual-norm kernels/actual decode, 1,636 total; all-kernel
sum ~59.995 ms versus GPU span ~65.253 ms. Those profiled values are diagnostic,
not an E2E speedup estimate. h48/h49 showed kernel removal alone did not deliver
the required >=1% E2E and regressed tails. This candidate tests a different
mechanism: dispatch gaps for the unchanged 11-node norm sequence, not fusion.
Two input copies plus one graph launch replace 11 individual host dispatches.
Only an unprofiled, admitted, counterbalanced comparison can prove any gain.

## Boundary

- ORIGINAL RMSNorm.forward_native callable, registered batch-invariant mean,
  BF16 intermediate rounding and every GPU arithmetic operator unchanged.
- BF16 contiguous [M,5120] residual case, M=1..64, ordinary norm flags only.
- Catalog per module, exact shape/stride/device/dtype, caller stream, weight
  address/version and epsilon. Inputs/outputs/graphs strongly retained.
- Separate private pools; no shared-pool clobbering assumptions. Same-stream
  FIFO orders copies/replays/consumers. Outputs are persistent and must not be
  retained past the next replay of the same module/key without a clone.
- Two eager side-stream warmups then capture. Synchronization/capture only in
  excluded initialization; seal before any steady request; new geometries fall
  back eagerly. Capture failure is terminal, not a silent measurement retry.
- No attention, FlashInfer plan, page-table, metadata, KV write, GEMM, sampler,
  scheduling or acceptance mutation. Framework graph configuration stays OFF;
  the experimental norm-only subgraphs are explicitly attested separately.
- Existing ragged full-target graph exact-geometry guard untouched.

## Gates

GPU unit bit parity, input immutability, consumer-before-reuse and sealed-miss
tests; actual service tokens/five acceptance fields against frozen h32 oracle,
serial and two concurrent waves at c1/c4/c8; two diagnostic nsys requests; then
fresh unprofiled baseline/baseline admission and 64-request A-B-B-A.
Every order requires >=1% mean E2E and zero TTFT/TPOT/E2E p95 regression.
Stop immediately on any failed gate, no request filtering/retry/matrix expansion.
Measured scope is p2048 c1 output128 only, not deployment or full matrix.

## Sources

- https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs
  Persistent addresses, output lifetimes, stream ordering, and private graph
  pools. Consulted current docs; installed runtime is pinned Torch 2.13.
- https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell/
  Separate host dispatch gaps from device work; do not capture host attention
  plans. Their Ling TP4 measurements are not evidence for our Qwen TP1 gains.

Performance: **UNPROVEN** until all prescribed gates pass.

## h51 initialization repair

First service never reached a request: target and draft register the same hook
factory in one process, and the first implementation incorrectly rejected the
second registration. Service stopped, no GPU process remained, serving source
unchanged. Factory is now idempotent for identical opt-in/config identity;
conflicting identities still fail closed. This is an initialization bug repair,
not a retry of a failed timing/correctness comparison. h51 logs retained.

## Final decision: rejected, h53

Fresh baseline/baseline noise gate passed (TPOT mean spread 0.0668%, E2E
0.0570%). First unprofiled AB, p2048 c1 output128, 64 steady samples/arm:

| Metric (ms) | Baseline mean / p95 | Graph mean / p95 |
|---|---:|---:|
| TTFT | 255.623 / 257.590 | 260.748 / 264.225 |
| TPOT | 18.9827 / 19.1635 | 18.9291 / 19.2012 |
| E2E | 2666.608 / 2691.114 | 2664.919 / 2847.383 |

Mean E2E improvement **0.0634%**, below 1%; p95 TTFT +2.5756%, TPOT +0.1969%,
E2E +5.8069%. Stop immediately; BA not run. All 450 saved requests matched
frozen tokens and five acceptance fields, including 117 serial/concurrent
correctness checks. Candidate captures stayed 800 after seal; replays increased
169550 -> 403550. Device-wide memory sampled 57005 -> 62711 MiB (+5706 MiB),
not allocator peak. All sampled clock-limit reasons zero. Tail root cause is
**unproven**; no GC/clock/queue causal claim and no discarded outliers.

Serving baseline remains detached 0577109; main unchanged; 8357 untracked
paths/list SHA unchanged. Owned candidate service stopped. Default opt-in OFF.
nsys node count stayed 1636/decode and norm 990: this confirms unchanged kernel
arithmetic, not E2E benefit. Do not retry this rejected timing comparison.
