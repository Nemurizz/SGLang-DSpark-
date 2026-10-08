# Concurrency measurement, inference code unchanged

h59 short barrier waves failed baseline c8 warmup admission: all 203 requests
matched immutable tokens/five acceptance, but two-wave group means did not
stay within 0.5%. No candidate timing ran. Saved logs remain intact; tag
`dspark-concurrency-short-wave-warmup-stopped-20261008`. No claim of graph gain
or regression can be made from h59. Active warmup clock-event reasons were zero,
but core clocks varied 2295–2422 MHz. This does not prove the cause of jitter.

Official SGLang guidance distinguishes steady online serving from a single
clearing batch and recommends at least five times concurrency in requests:

- https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx
- https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx

h60 is a separately prespecified CONTINUOUS-replenishment workload, not a
retry of h59 or a reclassification of its failure. Fixed client concurrency
4/8/16; clients immediately submit their next request on completion, no
per-iteration barrier. Each arm has three fixed 8-request-per-client warmup
blocks (mean spread <=0.5%), then 32-request-per-client measurement windows.
Whole windows include start/drain equally: no discarded timing rows.
Within-warmup request percentile ratios are not used as stationarity tests
because asynchronous concurrent client histories differ; this is recorded
explicitly, not silently substituted into the old serial/barrier protocol.

Fresh two-window baseline noise gates remain mean<=0.5%, p95<=1% for TPOT/E2E.
A-B-B-A. Candidate TTFT/TPOT/E2E p95 tolerance remains ZERO. Stop remaining
orders after a failing paired block; <1% mean alone still records BA if safe.
Every request checks frozen h32 output tokens and all five acceptance fields.
c16 cycles the eight existing frozen c8 prompt variants using slot%8; repeat
prompts cannot hit cache because both arms force full prefill/cache OFF.
This mapping is explicit, not a newly generated c16 reference.

Existing target graph remains STRICT bs1/[8]. Original geometry guard called
unchanged, and high/multirow geometry remains eager. Buffer/capture policy,
numerics/GEMM/reductions/FA4 fixed64/SPS/quantization are NOT changed. Original
graph-OFF service checkout and main stay untouched. Measurement is unprofiled.
No throughput/saturation/SLO/c32 claim may be inferred from capped c4/c8/c16.

## h60 stop and h61 one-time descriptive probing

h60 saved 167 frozen-correct requests, including two c16 bursts using explicit
slot%8 references. Continuous c4 fixed warmup mean TPOTs were 17.71060994,
17.53681484, 17.68512579 ms (0.991030% spread): qualification FAILED, and no
candidate timing ran. Both h59 and h60 were independently audited and archived
BEFORE changing any benchmark script. Their source copies remain in the archive.

The human request is to MEASURE concurrency, not necessarily to qualify a new
gain. Repeated qualification attempts are not useful at roughly 1% observed
baseline variability. h61 is ONE descriptive continuous c4/c8/c16 A/B. It has
an explicit `exploratory_descriptive_only` protocol flag and NEVER qualifies a
stable benefit, regardless of observed ratios. The same three fixed warmup
blocks and two baseline windows remain; FAILED warmup/noise flags stay FAILED,
not silently admitted. No request/tail filtering, retry, clock locking or new
inference policy. Candidate p95 tolerance stays zero; stop remaining orders on
any p95 regression, token/five-acceptance difference or illegal access. No more
protocol attempts will follow this run. Output includes caveated raw measured
TTFT/TPOT/E2E/p95/throughput and memory; no >=1% claim for high concurrency.

## h61 result: baseline latency stall; STOP, candidate timing never began

Serving configuration inherits `max-running-requests=8`. Thus c16 means 16
client requests (8-running cap plus queue), NOT GPU batch16. Peak client
concurrency was independently checked as 4/8/16; average in-flight includes
the final drain and is lower. This is not a capacity expansion experiment.

Measurement commit: a362c03. Baseline controls completed two windows at each
cell (32 requests/client/window); all failed the original noise admission.
c4 E2E mean/p95 spreads 0.9301%/1.1782%; c8 0.3212%/2.4278%; c16
22.6624%/6.0514%. Warmup/noise failures remain recorded as FAILED.

| Clients | Window | TTFT mean ms | TPOT mean ms | E2E mean ms | E2E p95 ms | E2E p99 ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 617.98 | 17.909 | 2892.59 | 3195.59 | 3752.82 | 160.08 |
| 4 | 1 | 604.44 | 17.806 | 2865.94 | 3158.38 | 3723.17 | 161.01 |
| 8 | 0 | 754.13 | 23.000 | 3675.25 | 5017.14 | 5564.21 | 238.70 |
| 8 | 1 | 751.13 | 22.931 | 3663.48 | 4898.22 | 5429.00 | 241.39 |
| 16 | 0 | 5274.00 | 29.264 | 8990.77 | 9041.83 | 67023.40 | 215.01 |
| 16 | 1 | 4299.82 | 23.856 | 7329.69 | 8525.89 | 9032.84 | 266.84 |

These are baseline-only DESCRIPTIVE measurements, NOT graph A/B results.
No long-tail requests were removed. 22:28:22–22:29:22 UTC+8: the engine log
has a 60-second gap after a 3-request/6144-token prefill with 5 running and
5 queued requests. GPU telemetry continued (299 samples, 294 at zero GPU
utilization; maximum entire-run sample gap0.251s; clock event reasons zero
throughout this interval). Fifteen requests took >60s. This suggests that
useful GPU work stalled while requests existed, but does NOT prove the CPU,
JIT, scheduler, driver, or kernel cause. No profiler/backtrace was collected.

On confirming concurrent latency instability, supervisor SIGTERM stopped the
verified OWNED driver (PID30194); its finally block drained owned services.
The first baseline-control arm finished; a new baseline arm had only six
checked correctness requests. The target_graph arm NEVER started; no AB or
BA comparison exists. It is therefore wrong to claim either concurrency
graph success OR graph-induced regression. Benefit remains UNPROVEN.

2541 h61 requests independently rechecked, including all 1792 measured
requests; all frozen tokens, five acceptance fields and zero retractions.
Together with h59/h60, all2911 saved requests rechecked. Steady acceptance
identical across the two baseline windows at every cell (c4 rate0.38469388;
c8/c16 rate0.38402458); do not compare these to c1's different prompt mix.
Device-wide memory sampled57007 MiB across all steady cells (NOT allocator
peak). Sources/oracles/tools intact; serving HEAD0577109 and main0c1f1eb
unchanged; 8357 existing untracked paths retained. GPU process list empty.
Old h58 c1 gain remains separately scoped, not extrapolated to concurrency.
No more protocol retries or optimization experiments were opened.
