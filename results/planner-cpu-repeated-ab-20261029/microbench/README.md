# Same-process planner microbenchmark

The cached and uncached calls reuse the same `SpsAdditiveCostTable`, the same `DSparkScheduleConfig`, and the same survival tensor for every pair. Calls are alternated in one Python process; n=10,000 per path/request-count. Times are microseconds.

| request count | uncached mean/std/p95/p99 | cached mean/std/p95/p99 | reduction | budget equal |
|---:|---|---|---:|---|
| 1 | 87.537 / 12.911 / 90.467 / 143.921 | 39.147 / 14.148 / 40.717 / 45.011 | 55.28% | True |
| 4 | 89.432 / 7.226 / 94.823 / 122.053 | 39.842 / 25.079 / 42.004 / 63.803 | 55.45% | True |
| 8 | 91.872 / 20.870 / 99.304 / 118.414 | 40.523 / 5.013 / 43.796 / 62.902 | 55.89% | True |
| 16 | 95.721 / 2.198 / 97.790 / 102.005 | 41.493 / 1.907 / 42.981 / 44.957 | 56.65% | True |

Each mean has a bootstrap 95% CI in `same_process_microbench.json`.
