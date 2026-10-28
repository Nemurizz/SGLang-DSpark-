# CPU planner cache A/B on semantic serving (2026-10-29)

This run isolates the planner cost-vector cache in the dynamic DSpark path. `dynamic_uncached` sets `SGLANG_DSPARK_DISABLE_PLANNER_COST_CACHE=1`; `dynamic_cached` uses the optimized cache (default). AR and full verify are included as context.

## Fixed protocol

- Same 12 fixed semantic prompts as the earlier suite: 4 code, 4 math, 4 open QA; tokenizer shape 512 input tokens and 128 output tokens.
- Same arrival schedule: 2.0 requests/s, 0.5 s interval; c=1/4/8; 12 requests per cell.
- Same Qwen3-14B BF16 server, draft, sampling, seed, request order, cache/FP8/CUDA Graph switches, and server lifecycle.
- Only dynamic planner cache mode changes between `dynamic_uncached` and `dynamic_cached`; the new env flag bypasses only `_additive_step_time_cache`.
- Server records include planner CPU, CPU/GPU step times, budget, accepted length, acceptance, TTFT, TPOT and E2E.

## Direct before/after result

All cells are mean / p95 unless marked otherwise. CPU and latency units are milliseconds.

| c | metric | uncached | cached | mean change |
|---:|---|---:|---:|---:|
| 1 | planner CPU mean / p95 | 0.307 / 0.459 | 0.191 / 0.254 | +37.72% |
| 1 | CPU step mean / p95 | 64.93 / 74.27 | 65.74 / 74.74 | -1.25% |
| 1 | GPU step mean / p95 | 64.92 / 70.83 | 65.72 / 70.68 | -1.22% |
| 1 | TTFT mean / p95 | 3850.5 / 6826.4 | 3894.1 / 6934.6 | -1.13% |
| 1 | TPOT mean / p95 | 8.18 / 8.43 | 8.29 / 8.48 | -1.33% |
| 1 | E2E mean / p95 | 4889.3 / 7825.3 | 4946.6 / 7994.8 | -1.17% |
| 4 | planner CPU mean / p95 | 0.309 / 0.398 | 0.184 / 0.219 | +40.65% |
| 4 | CPU step mean / p95 | 62.30 / 72.16 | 61.09 / 65.17 | +1.94% |
| 4 | GPU step mean / p95 | 62.62 / 69.96 | 61.20 / 65.12 | +2.27% |
| 4 | TTFT mean / p95 | 988.5 / 1772.4 | 835.9 / 1614.6 | +15.44% |
| 4 | TPOT mean / p95 | 9.88 / 9.17 | 9.54 / 8.78 | +3.47% |
| 4 | E2E mean / p95 | 2243.0 / 3241.0 | 2046.9 / 3066.6 | +8.75% |
| 8 | planner CPU mean / p95 | 0.305 / 0.357 | 0.197 / 0.333 | +35.17% |
| 8 | CPU step mean / p95 | 61.43 / 67.58 | 60.76 / 65.18 | +1.09% |
| 8 | GPU step mean / p95 | 61.60 / 65.39 | 61.26 / 65.10 | +0.56% |
| 8 | TTFT mean / p95 | 567.3 / 1681.1 | 554.6 / 1647.5 | +2.23% |
| 8 | TPOT mean / p95 | 9.95 / 9.84 | 9.84 / 9.75 | +1.05% |
| 8 | E2E mean / p95 | 1830.8 / 3227.9 | 1804.9 / 3181.4 | +1.41% |

## Interpretation

1. The full serving planner CPU mean falls from 0.3066/0.3093/0.3045 ms to 0.1909/0.1836/0.1974 ms at c=1/4/8: 37.7%, 40.7%, and 35.2% lower, respectively. The p95 reduction is 44.6%, 44.9%, and 6.9%; the c=8 tail is noisy at n=68–69 planner observations.
2. The optimized cache does not change the verifier acceptance in this run: both variants are 1.000 for all three cells. Budget distributions are close but not byte-identical across separate server processes (especially c=4), so E2E is an observed service comparison rather than a pure causal estimate of cache cost.
3. Observed E2E mean changes are +1.17% (c=1), -8.75% (c=4), and -1.42% (c=8) when moving from uncached to cached. The c=4 improvement is much larger than the planner CPU saving alone and should not be attributed entirely to the cache; separate processes, GPU scheduling noise, and slightly different dynamic budgets contribute.
4. The planner cache saves only about 0.11–0.13 ms per decision in the full semantic path. It is therefore a meaningful CPU-path optimization, but it is not a standalone 2x end-to-end serving optimization. The earlier microbenchmark measured only additive-vector construction and showed 2.20x; this service A/B measures the entire `compute_budget` call.

## Reproduction and evidence

- `run_planner_cpu_semantic_ab.py`: runner with the uncached/cached flag.
- `planner_cache_toggle.patch`: source change adding the opt-in uncached switch; default behavior remains cached.
- `planner_ab_summary.json`: paired deltas.
- `summary.json`, `*_c*/result.json`, `*_c*/server.log`: raw records and logs for all 12 cells.
- `server_suite.log`: runner output.

## Limitations

- 12 responses per cell and separate server processes are a baseline sample, not a confidence interval.
- Prompt shape is fixed and acceptance is saturated at 1.0; this measures planner/service overhead, not semantic quality.
- Because GPU execution and dynamic confidence can vary across fresh processes, use repeated batches or an in-process paired harness for a final causal estimate of E2E delta.

## Microbenchmark context

The focused additive-vector microbenchmark from the CPU planner retest measured the same cached function in isolation:

| request count | uncached | cached | reduction | speedup |
|---:|---:|---:|---:|---:|
| 1 | 84.322 us | 38.667 us | 54.1% | 2.18x |
| 4 | 87.162 us | 39.366 us | 54.8% | 2.21x |
| 8 | 88.272 us | 40.169 us | 54.5% | 2.20x |
| 16 | 94.242 us | 42.583 us | 54.8% | 2.21x |

The isolated benchmark is not the same as the full `compute_budget` serving measurement: it times only additive step-time vector construction. The service A/B above times the whole planner call, which is why its measured CPU reduction is about 35-41% rather than 54-55%.
