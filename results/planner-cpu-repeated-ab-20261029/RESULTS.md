# Repeated CPU planner cache A/B (2026-10-29)

This is the repeated service-level follow-up. Each concurrency has five independent uncached cells and five independent cached cells, 12 requests per cell. The 30 cells were shuffled with seed 20261029; each cell used the same prompt order and 0.5-second request schedule. `cached_minus_uncached` uses the matching trial index as a pair.

## Fixed protocol

- 12 semantic prompts: 4 code, 4 math, 4 open QA; 512 input tokens, 128 output tokens.
- Qwen3-14B BF16; cache/radix, FP8 and CUDA Graph disabled; same draft, seed, sampling and request order.
- Arrival rate 2 req/s; c=1/4/8; five independent server runs for each planner mode and concurrency.
- `dynamic_uncached`: `SGLANG_DSPARK_DISABLE_PLANNER_COST_CACHE=1`; `dynamic_cached`: default cache.
- A same-process microbenchmark reuses one survival tensor and alternates uncached/cached calls 10,000 times per request count.

## Service-level distributions

Each distribution cell is `mean / std / p95 / p99; bootstrap 95% CI for the mean`. Latency/CPU units are ms.

| c | metric | uncached | cached | paired cached−uncached mean [bootstrap CI] | reduction |
|---:|---|---|---|---:|---:|
| 1 | planner CPU | 0.281 / 0.172 / 0.368 / 0.511; CI [0.272, 0.293] | 0.185 / 0.048 / 0.264 / 0.369; CI [0.182, 0.188] | -0.096 [-0.113, -0.081] | +34.28% |
| 1 | CPU step | 65.02 / 5.34 / 73.92 / 80.19; CI [64.69, 65.36] | 64.73 / 5.04 / 73.72 / 76.89; CI [64.42, 65.05] | -0.29 [-1.41, 0.85] | +0.44% |
| 1 | GPU step | 65.11 / 2.74 / 69.78 / 74.89; CI [64.94, 65.28] | 64.76 / 2.47 / 69.26 / 72.90; CI [64.61, 64.91] | -0.35 [-1.51, 0.84] | +0.53% |
| 1 | TTFT | 3838.6 / 2326.8 / 7339.2 / 7661.3; CI [3258.0, 4400.8] | 3811.5 / 2309.7 / 7405.6 / 7543.5; CI [3237.0, 4372.2] | -27.1 [-123.9, 76.6] | +0.71% |
| 1 | TPOT | 8.19 / 0.17 / 8.47 / 8.65; CI [8.15, 8.24] | 8.16 / 0.20 / 8.43 / 8.46; CI [8.10, 8.20] | -0.04 [-0.18, 0.11] | +0.46% |
| 1 | E2E | 4879.1 / 2314.8 / 8351.9 / 8693.0; CI [4324.9, 5461.7] | 4847.2 / 2296.0 / 8397.5 / 8564.4; CI [4298.2, 5428.7] | -31.9 [-143.9, 92.6] | +0.65% |
| 4 | planner CPU | 0.293 / 0.068 / 0.430 / 0.524; CI [0.286, 0.300] | 0.219 / 0.213 / 0.335 / 0.603; CI [0.202, 0.247] | -0.074 [-0.098, -0.051] | +25.37% |
| 4 | CPU step | 61.79 / 4.77 / 70.80 / 74.60; CI [61.25, 62.35] | 62.34 / 6.11 / 72.59 / 82.13; CI [61.64, 63.10] | 0.53 [-0.75, 1.71] | -0.86% |
| 4 | GPU step | 61.99 / 3.38 / 67.38 / 74.40; CI [61.62, 62.36] | 62.48 / 4.35 / 69.70 / 76.57; CI [62.02, 62.98] | 0.49 [-0.61, 1.59] | -0.79% |
| 4 | TTFT | 898.0 / 597.5 / 1950.9 / 2061.9; CI [747.2, 1055.6] | 1101.1 / 726.3 / 2338.2 / 2790.9; CI [916.3, 1293.3] | 203.1 [-44.5, 617.7] | -22.62% |
| 4 | TPOT | 9.73 / 4.32 / 23.35 / 24.48; CI [8.73, 10.84] | 9.94 / 4.85 / 23.74 / 27.44; CI [8.84, 11.21] | 0.21 [-0.15, 0.63] | -2.20% |
| 4 | E2E | 2133.1 / 659.0 / 3227.7 / 3241.9; CI [1965.4, 2300.0] | 2363.4 / 763.5 / 3490.6 / 4160.0; CI [2170.5, 2554.6] | 230.3 [-64.1, 705.8] | -10.80% |
| 8 | planner CPU | 0.291 / 0.058 / 0.405 / 0.482; CI [0.285, 0.297] | 0.207 / 0.171 / 0.328 / 0.447; CI [0.193, 0.227] | -0.084 [-0.114, -0.046] | +28.76% |
| 8 | CPU step | 61.89 / 5.15 / 70.71 / 75.18; CI [61.31, 62.47] | 61.92 / 4.76 / 70.09 / 75.72; CI [61.38, 62.47] | 0.04 [-1.04, 1.13] | -0.06% |
| 8 | GPU step | 62.13 / 3.57 / 67.76 / 70.60; CI [61.75, 62.50] | 62.13 / 3.67 / 67.54 / 73.63; CI [61.73, 62.53] | 0.01 [-0.91, 0.89] | -0.02% |
| 8 | TTFT | 598.3 / 673.6 / 1955.4 / 2101.3; CI [436.6, 774.8] | 590.2 / 667.4 / 1953.1 / 2093.6; CI [428.9, 766.1] | -8.1 [-57.0, 40.8] | +1.36% |
| 8 | TPOT | 9.94 / 4.59 / 24.37 / 25.49; CI [8.89, 11.25] | 10.01 / 4.60 / 24.47 / 25.76; CI [8.96, 11.32] | 0.06 [-0.08, 0.21] | -0.65% |
| 8 | E2E | 1861.1 / 819.0 / 3352.4 / 3387.3; CI [1667.1, 2065.9] | 1861.1 / 820.3 / 3337.2 / 3436.3; CI [1664.9, 2067.0] | 0.0 [-64.9, 67.4] | -0.00% |

## Acceptance and budget

| c | uncached acceptance | cached acceptance | uncached budget mean/p95 | cached budget mean/p95 |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 7.00 / 7.00 | 7.00 / 7.00 |
| 4 | 1.000 | 1.000 | 21.44 / 28.00 | 22.04 / 28.00 |
| 8 | 1.000 | 1.000 | 21.48 / 49.00 | 21.42 / 49.00 |

## Interpretation

1. The same-process microbenchmark is the cleanest CPU result: mean uncached-to-cached reduction is 55.5%–56.7% for request counts 1/4/8/16, all paired budgets are equal, and the JSON includes standard deviation, p95, p99 and bootstrap mean intervals.
2. The repeated service A/B gives five independent samples per mode/concurrency. Planner CPU reduction is estimated from paired trial means; the service table reports both distribution statistics and a bootstrap interval for cached−uncached.
3. E2E must still be interpreted with the paired intervals and budget traces. Separate server processes can follow slightly different confidence/budget trajectories; the repeated design quantifies that spread instead of hiding it behind one run.

## Reproduction files

- `run_repeated_ab.py`, `cell_order.json`, `summary.json`: randomized 30-cell runner and order.
- `aggregate.json`, `paired_deltas.json`: all distributions and paired bootstrap results.
- `microbench/`: same-process survival-tensor benchmark and its statistics.
- `*_c*_r*/`: 30 raw result JSON/log directories.
- `server_suite.log`: full runner log.

## Statistical notes

- Bootstrap intervals resample the observed values 5,000 times with replacement; they are descriptive for this finite run set.
- p95/p99 are empirical quantiles. For 60 response observations per mode/cell they are still noisy; use more repetitions for a production SLO claim.
- The paired effect uses trial index pairing; execution order was shuffled independently of the mode label.
