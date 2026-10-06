# Validation decision, 2026-10-07

The eager serial-reference attention candidate is validated for a limited
full-prefill correctness smoke, NOT adopted as a cache performance optimization.
No GEMM, graph or quantization experiments were performed. Defaults stay OFF.

## Checked implementation

Inference change: `48b5d0c` on `exp/verify-serial-reference-plans`.
Frozen-oracle gate: `e6d4105`; fresh HTTP policy/tests: `5f52c82`, `73d6a81`.
Tested tag: `dspark-verify-reference-plans-full-prefill-smoke-20261007` at
`73d6a817d32db02205d394aa512ddf7e6a7160bf`. No merge or push.

The checked FA2 0.6.17 eager planner used 128-token KV chunks for B1 and 412 for
B8 at the first original 2K mismatch. h13 captured identical layer-0 Q/K/V,
past KV and current KV writes, but 1955/5120 changed attention-output elements
(max absolute difference 0.0009765625). CPU comparison checks saved tensor SHAs.
Focused attention replay exactly reproduced both automatic-plan outputs. Fixed
128 reproduced original B1 there; fixed 4096 changed it. This is correctness
evidence, not a kernel latency benchmark.

The candidate preserves EACH request's original B1 KV chunk and CTA query-tile
rule, groups adjacent equivalent geometries, preserves row order and rejects
unsupported/stale/padded layouts. B1 retains the original automatic wrapper.
It does not relax exact graph guards or deterministic prefix-cache restrictions.

## Evidence and limits

| Run | Purpose | Result |
| --- | --- | --- |
| h12 | Upstream observer first run | Diagnostic FFI Array serialization error; stopped/restored; no model conclusion |
| h13 | Corrected five-stage observer | Original mismatch reproduced, 15 snapshots, first differing boundary identified |
| h14 | Candidate, 2K C8, independent ordinary-target oracle | 56 completed requests; three serial/concurrent rounds match tokens and DSpark acceptance |
| h15 | Original a61e0bb, missing 8K serial oracles | 13 new references each repeated consistently; combined original oracle has 52 variants across 12 cells |
| h16 | First complete candidate matrix | Nine cells completed; 8K C1 transport disconnect; stopped/restored, no preceding model/CUDA exception in log |
| h17 | Same frozen oracle, fresh HTTP, no generation retries | All 12 cells completed; 200 request records, 32 output tokens each; tokens and acceptance match original references |
| h18 | Cache-compatible full/paired A/B candidate | Stopped at first FULL reference, p128_c1 token index 2: frozen 34334 vs actual 16555; paired never started |
| h19 | Original a61e0bb, candidate OFF, same failed profile | Original output/acceptance exactly match h18; the profile also differs from frozen deterministic oracle |

h17 covers `128/512/2048/8192 x 1/4/8`, four warmups excluded from evidence,
52 cold references + 52 prime requests + 96 steady smoke requests. Each cell
has only eight measured samples and one repeat. This is NOT sufficient-sample
p95, long-output, multi-seed, counterbalanced steady A/B or production validation.
The independent ordinary-target oracle covers the specific h14 2K C8 case;
the whole h17 matrix is checked against original DSpark serial references.

56 unit/capability tests passed (+ three subtests), including existing ten
ragged-layout and two backend-capability tests. Deprecated torch.jit warnings
do not change that result. The first h16 transport-cause hypothesis remains a
hypothesis; one successful fresh-connection rerun does not prove its root cause.

## Cache/performance decision

Current deterministic FlashInfer configuration disables radix caching by the
existing SGLang guard. We did not remove it. The cache-compatible non-deterministic
profile changes original B1 output before ANY paired-cache reuse. h19 separates
that existing profile difference from a candidate-only regression, not a proof
that the paired cache itself is faulty or useless.

No valid full/target-only/paired comparison has completed under the frozen
output contract. TTFT/TPOT/p95/memory in h17 are smoke control observations only.
Shadow hit, paired hit, partial recovery and actual draft-fallback outcomes are
unavailable, not zero; no cache or end-to-end speedup is proven. Do not launch
target-only shadow or loosen the oracle to bypass this failure.

A separate original cache-compatible oracle would require an explicit reference
policy decision. Preserve the current deterministic reference and label profiles
separately; do not silently replace it with new candidate-generated references.

## Rollback

All owned services were stopped and checkout restored to
`exp/reuse-compact-index@a61e0bb` (`dspark-optimization-history`). Original 8357
untracked entries were not deleted. main remains 0c1f1eb. Candidate changes remain
on their separate branch/tag only. Restoring a61 recovers the user's checkpoint;
it does NOT imply a61 passed the newly expanded concurrent correctness tests.
