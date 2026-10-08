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
