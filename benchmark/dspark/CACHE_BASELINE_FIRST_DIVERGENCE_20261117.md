# Paired cache baseline divergence diagnosis 2026-11-17

## Result

The earlier cache-compatible baseline gate stopped at `p128/c4` because concurrent variant 0 produced token `16555` at the third output position while its old reference expected `34334`; correct drafts changed from 22 to 21. The first interpretation was “the baseline is invalid,” which is correct, but the cause needed narrowing.

The stored manifests show that this gate was not using the later qualified numerical profile:

| Run | Commit | Output | Numerical profile | Status |
|---|---|---:|---|---|
| `dspark-cache-compatible-original-baseline-20261007-h21` | `a61e0bb` | 32 tokens | no fixed FA4 deterministic profile recorded | safety stopped |
| `dspark-stable-fa4-fixed64-long128-20261007-h32` | `0577109` | 128 tokens | deterministic FA4 prefill/verify, fixed 64×64, BF16 torch GEMM, graph off | full/paired correctness passed for requested cells |

The h21 data itself shows the mismatch is tied to the old protocol and prompt/batch mapping: in the same `p128_c4` reference set, variant 0 starts with `34334`, while variants 1–3 start with `16555`. During the failed concurrent block, variant 0 produced the variant-1 style token. That is enough to reject the old reference comparison, but it does not prove a paired-cache bug.

## Diagnosis

The failed gate compared a new concurrent execution against an oracle produced under a different commit and numerical policy. The old run had no fixed FA4 deterministic profile in its manifest and used a 32-token smoke contract; the later qualified run froze a separate 128-token reference under `0577109` and `dspark-stable-fa4-fixed64-v2`. The later status records both full and paired correctness as passed for its requested cells.

The first divergent position is still useful: it identifies the shape/batch-sensitive path to inspect if a fresh baseline diverges under the qualified profile. It cannot be used to measure cache performance because the oracle, source head, output contract and profile were not aligned.

## Decision

This task is complete. The next cache performance experiment must start from the qualified `0577109` / fixed64 / 128-token reference, use the same request-to-variant mapping, and stop immediately if the first concurrent cell diverges. Do not reuse the h21 `a61e0bb` reference for paired-cache A/B.

Evidence paths on the remote machine:

- `/root/autodl-tmp/results/dspark-cache-compatible-original-baseline-20261007-h21`
- `/root/autodl-tmp/results/dspark-stable-fa4-fixed64-long128-20261007-h32`
- `/root/autodl-tmp/results/dspark-cache-natural-recovery-20261007-h33`
