# Paired cache correctness and recovery status 2026-11-16

## This two-hour run

The first task was the paired-cache correctness closure. The CPU-only protocol tests passed: 22 DSpark A/B protocol tests and 5 stable-cache validation tests. The real GPU original-cache baseline was then checked against the independent stable FA4 reference. It stopped safely at `p128/c4`, concurrent phase, variant 0, request token 3: the reference token was `34334`, while the concurrent run produced `16555`; correct drafts changed from 22 to 21. No paired performance run was started from this invalid baseline.

The second task was the natural partial-recovery pressure gate that already existed in the remote evidence directory. Its manifest uses the stable FA4 fixed64 profile, Qwen3-14B target, DSpark block7 draft, BF16 target/KV, graph off, and the 25% sidecar metadata budget. It is a correctness and execution-coverage result, not a timing A/B.

## Natural partial recovery result

| Metric | Result |
|---|---:|
| Completed requests | 136 / 136 |
| Tested concurrencies | 1 / 4 / 8 |
| Match calls | 128 |
| Target hit tokens | 999302 |
| Paired hit tokens | 703786 |
| Metadata evictions | 225 |
| Actual partial-recovery requests | 40 |
| Actual recovered tokens | 640 |
| Actual partial-prefill tokens | 680 |
| Actual draft-injected tokens | 680 |
| Actual target-only full-fallback requests | 36 |
| Partial-recovery coverage | 640 tokens across 40 requests |

The important point is that recovery was observed in worker records, rather than inferred from a scheduler plan counter. The recovered token count and draft injection count were both 680 at the worker instrumentation boundary, and every request joined successfully. The stage passed its requested recovery gate, but its status remains `passed_for_requested_stage_only` and its performance claim remains `unproven`.

## What can be concluded

The sidecar can survive natural metadata pressure and distinguish several states: a complete paired prefix, a target hit with a missing draft suffix, and a target-only full fallback. In the partial case, the scheduler preserves the paired prefix and recomputes only the missing suffix. In the full-fallback case, it does not pretend that a target-only hit is safe for DSpark.

The result also shows why the metadata budget matters: 225 metadata evictions occurred while the GPU KV capacity remained sufficient. This is evidence that target KV residency and draft-readiness metadata have different lifecycles. It is not evidence that the cache is faster in production.

## Current limitations

- The original baseline gate is not valid for the full correctness matrix because concurrent `p128/c4` diverged from the independent reference. A new paired performance A/B must wait for a clean numerical baseline.
- The recovery pressure gate used a fixed synthetic pressure construction. It did not represent a production prefix-length or arrival-time distribution.
- The result did not measure TTFT, TPOT, E2E, throughput, or cache prime amortization.
- The correct target-only plus draft-recovery control is still missing, so paired cache cannot yet be claimed better than that alternative.
- The result covers the requested recovery stage and concurrencies 1/4/8; it does not cover the full 128/512/2048/8192 correctness matrix or long-output performance.

## Next experiment

Before starting the remaining cache experiments, fix and requalify the numerical baseline at the first divergent shape. The next debugging target is the concurrent `p128/c4` execution path: capture the first divergent token, verify the FA4 geometry and batch shape, and compare the target hidden/logit tensors before changing cache policy. If the baseline cannot pass, cache performance numbers should remain blocked.

Evidence was produced by the remote directories `dspark-cache-compatible-original-baseline-20261007-h21` and `dspark-cache-natural-recovery-20261007-h33`; the public repository contains this summary and the paired-cache protocol.
