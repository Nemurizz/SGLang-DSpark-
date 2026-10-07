# Authorized original cache-compatible baseline gate (h21)

The human explicitly authorized a separate original cache-compatible reference,
preserving the deterministic reference unchanged. This run attempted that gate;
it did NOT run paired cache, GEMM, graphs, quantization, or a performance A/B.

Runtime was clean tracked `a61e0bb`, original full-prefill DSpark configuration,
deterministic OFF, all graphs OFF, candidate OFF, PyTorch greedy, seed 20261007,
32 output tokens and the existing 12-cell dataset. Radix structures remained
enabled exactly as h19; the original DSpark key_limit=0 guard prevented reuse.
All four warmups and all 19 recorded requests had cached_tokens=0.

The driver independently collects each cell's cold serial references, then checks
three rounds of serial and concurrent responses token-by-token and against all
five acceptance fields. It publishes references.jsonl ONLY after every cell
passes. The old deterministic file is read-only and its SHA is checked again
after stopping the service. The p128_c1 reference must also match original h19.

## Safety stop

- p128_c1 completed three serial/concurrent rounds.
- p128_c4 collected four cold references and completed one serial round.
- Its FIRST concurrent round failed for variant 0, zero-based output index 2:
  serial reference token **34334**, concurrent token **16555**.
- Correct drafts also changed **22 -> 21**. This is a same-profile original
  B1/B4 difference, not a deterministic ON/OFF comparison.
- 19 raw requests were saved (5 references, 7 serial repeats, 7 concurrent
  requests; the p128_c4 batch finished near-simultaneously). Four warmups are
  excluded. Exactly one cell passed; the remaining matrix did not run.
- Complete references.jsonl was NOT published; partial raw rows cannot be used
  as the full matrix oracle. Paired service never started.
- Service stopped, port 30117 closed, GPU 0 MiB / 0%, original a61e0bb retained.
  The deterministic reference SHA was unchanged.

## Interpretation

Creating a separate reference policy does not make a non-batch-invariant
baseline stable. This original cache-compatible configuration currently fails
the required serving-correctness gate. Performance comparisons against it cannot
pass project acceptance. This does NOT prove paired cache has zero benefit or
that paired cache caused this mismatch; cache reuse was never active here.

The exact causal operator behind this B1/B4 case has not been isolated. Do not
attribute it to a specific GEMM, norm, prefill, or verify kernel without further
evidence. Do not bypass the deterministic/cache guard, waive acceptance fields,
or regenerate optimized references to hide the failure.

## Reproduction

Run from a separate copy of the harness files using the server Python environment
while the inference checkout is original clean tracked a61e0bb:

```bash
python cache_profile_correctness.py \
  --control-run /root/autodl-tmp/results/dspark-original-cache-profile-20261007-h19 \
  --output /root/autodl-tmp/results/dspark-cache-compatible-original-baseline-20261007-h21 \
  --repeats 3
```

The existing output directory is preserved evidence. Use a NEW output path for
another run. Driver/harness SHA, reference provenance, configuration, package
versions, token IDs, acceptance metadata and shutdown are recorded.

Initial new CPU unit tests: 4 passed locally and on the server. Final combined
regression: **66 tests passed, 3 subtests passed**, no skips on the server;
14 existing torch.jit deprecation warnings. This includes the previous 62 tests
and the four new publication/profile/cache/acceptance guards. The companion
PAIRED_CACHE_BENEFIT_PROTOCOL.md specifies the proof requirements and openly lists
unimplemented balanced-order/fallback-observability and workload coverage gaps.
