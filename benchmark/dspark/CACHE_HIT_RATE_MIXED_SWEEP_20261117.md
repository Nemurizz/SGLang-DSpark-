# Paired cache mixed hit rate sweep 2026-11-17

## Protocol

This is the third cache task. The repository does not contain a production request trace, so this run uses a deterministic mixed-prefix trace and is labeled synthetic. It uses the current qualified FA4 fixed64 profile (`0577109`), Qwen3-14B target, DSpark block7 draft, BF16 target/KV, graph off, `p512/c4`, output 128, 8 measured requests and 2 warmup batches per rate. Full-prefill and paired-cache use independent services with the same seeded prompt construction. Reuse rates are 0/10/25/50/75/100%.

The hit-rate percentage refers to measured requests that reuse one of four exact hot prefixes. Token-level hit rate is reported separately because the shared prompt template leaves a small structural prefix even in the 0% case.

## Result

| Reuse target | Paired token hit | Full E2E mean | Paired E2E mean | E2E change | Full p95 | Paired p95 | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0%   | 1.93% | 3598.46 ms | 3596.19 ms | +0.06% | 4940.00 ms | 4934.95 ms | 0.3443 |
| 10%  | 14.16% | 3585.21 ms | 3583.74 ms | +0.04% | 5001.22 ms | 4996.79 ms | 0.3417 |
| 25%  | 26.47% | 3760.02 ms | 4036.96 ms | **−7.37%** | 5044.37 ms | 5586.32 ms | 0.3265 |
| 50%  | 50.98% | 3646.81 ms | 3571.03 ms | +2.08% | 4966.38 ms | 4875.71 ms | 0.3402 |
| 75%  | 75.49% | 3801.50 ms | 3657.95 ms | +3.78% | 5224.13 ms | 5050.85 ms | 0.3238 |
| 100% | 100.00% | 3671.81 ms | 3632.12 ms | +1.08% | 4938.37 ms | 4915.90 ms | 0.3330 |

`E2E change` is `1 - paired/full`; positive values mean paired is faster. Every paired cell completed with one match invocation per request, paired coverage equal to target coverage, zero partial recovery, and zero full recovery. Queue wait was not collected as a separate field in this sweep; E2E includes queueing and service execution.

## Conclusion

At 0% and 10% reuse, the paired cache has no measurable E2E benefit in this run. At 25%, paired E2E became 7.37% slower and p95 increased by about 10.7%, showing that sidecar lookup, cache lifecycle and request scheduling costs can dominate when reuse is sparse. At 50% and 75%, the run showed 2.08% and 3.78% mean E2E reductions. The 100% hot-prefix case showed only 1.08% because the remaining decode, speculative verification and queueing time dominates after prefill is removed.

This does not establish a production threshold such as “50% is always enough.” Each rate has only 8 measured requests and one service pair, and the prompt trace is synthetic. The result establishes the direction of the tradeoff and the next measurement requirement: replay a real prefix-length and arrival-time trace with independent repetitions, cold/warm/mixed phases, and explicit queue-wait measurement.

## Task status

This task is complete. The next task should use a production-like trace or a larger deterministic pilot before changing cache admission or metadata policy. No partial-recovery claim is made by this sweep; all recovery counters were zero.

Remote output directory: `/root/autodl-fs/dspark-cache-hit-rate-20261117-current`.
