# Original deterministic/cache-compatible profile isolation (h20)

This is a read-only diagnosis, not a cache speedup or a new inference candidate.
No GEMM, CUDA graph, or quantization experiments were performed.

Both serving arms ran clean tracked `a61e0bb`, candidate OFF, all graphs OFF,
greedy PyTorch sampling, seed 20261007, the frozen p128_c1 variant 0 prompt,
32 output tokens, four p512_c1 warmups, and two serial measured repeats.
Actual cache reuse was zero for every request. The non-deterministic arm retained
the original h19 radix structures, but the existing DSpark key_limit=0 guard
prevented reuse. Its server_info.disable_radix_cache is therefore false; it must
not be misreported as globally disabled radix structures.

The deterministic arm checked tokens and all five acceptance metadata fields
against the original frozen h15 reference. The cache-compatible arm checked
against the saved original h19 diagnostic request, NOT a newly generated oracle.
The global deterministic oracle was not replaced or modified.

The original read-only hook captured the first committed verify anchor at
position 128, input token 576, B1, eight live queries. It did NOT inspect a
future proposal as if it were committed. Five stages per repeat were captured;
all ten captures per arm were complete and SHA-checked by a CPU-only comparison.

## Observations

Both repeats within each profile were bitwise equal at every captured tensor.
This is a two-repeat, one-prompt serial result, NOT a concurrency stability gate.

Cross-profile differences at layer 0:

| Tensor | Changed elements | Maximum absolute difference |
| --- | ---: | ---: |
| Query | 2063/5120 | 0.0625 |
| Current key | 427/1024 | 0.5 |
| Current value | 396/1024 | 0.0009765625 |
| Past key (128 prompt positions) | 43559/131072 | 1.0 |
| Past value (128 prompt positions) | 41868/131072 | 0.001953125 |
| Attention output | 3568/5120 | 0.0009765625 |

The actual written current K/V exactly matched call inputs within BOTH profiles.
The original target-verify wrapper plan metadata was identical across profiles.
Already-different Q/current KV/past KV rule out treating this as only the previous
B1/B8 verify-planner partition problem. They do not identify the exact earliest
upstream operator; no single RMSNorm, projection, or prefill operation has been
causally isolated by this run.

At the captured anchor, raw logits changed in 130592/151936 elements, max abs
0.25, while argmax stayed 2197. This snapshot is NOT the divergent third-output
row. The completed generation still reproduced the known third-token difference:
deterministic 34334 versus cache-compatible 16555 (zero-based output index 2).

Both profiles had ten verify calls and seventy proposed drafts; correct drafts
were 22 versus 21, with different acceptance histograms. Each profile's two
requests matched its preserved control tokens AND acceptance metadata.

## Source-level explanation and limits

Existing deterministic configuration enables batch-invariant mm/addmm/bmm,
mean, log-softmax and RMSNorm dispatch, changes RMSNorm execution, and changes
FlashInfer prefill from ragged to paged with a fixed split setting. It also
disables radix cache for this backend. Switching deterministic OFF is not a
cache-only treatment. These source-level differences explain why a cross-profile
cache A/B is confounded; the exact numerical first cause is still unproven.

The two arms use the original guarded profile commands. No guards were bypassed,
no kernels or arithmetic were replaced, and no latency was treated as performance
evidence. The diagnostic controls permit observing the already-known profile
differences, but do NOT authorize a paired cache performance run with a redefined
oracle. That reference-policy decision must be explicit.

## Reproduction

Use the server Python environment and original clean tracked a61e0bb checkout.
The harness/hook and these three scripts can reside outside the checkout; their
SHA256 values are recorded in each arm's manifest. Each arm refuses an existing
port/output folder and stops only its own process group in finally.

```bash
python trace_original_cache_profile.py \
  --control-run /root/autodl-tmp/results/dspark-original-cache-profile-20261007-h19 \
  --arm deterministic \
  --output /root/autodl-tmp/results/dspark-original-profile-trace-det-20261007-h20a

python trace_original_cache_profile.py \
  --control-run /root/autodl-tmp/results/dspark-original-cache-profile-20261007-h19 \
  --arm cache-profile \
  --output /root/autodl-tmp/results/dspark-original-profile-trace-cache-20261007-h20b

python compare_original_cache_profiles.py \
  --deterministic /root/autodl-tmp/results/dspark-original-profile-trace-det-20261007-h20a \
  --cache-profile /root/autodl-tmp/results/dspark-original-profile-trace-cache-20261007-h20b \
  --output /root/autodl-tmp/results/dspark-original-profile-comparison-20261007-h20.json
```

The reproduction paths are preserved existing evidence. Use NEW output paths
for another run; do not overwrite or remove the saved folders.

Initial observer/isolation unit checks: 13 tests OK. Final combined regression:
62 tests passed, 3 subtests passed, 14 existing torch.jit deprecation warnings.
This includes the original 56 regression tests and six new isolation/comparison
tests; none were skipped on the server. Services stopped at the end,
original a61e0bb retained, main unchanged, existing untracked files preserved.
Paired-cache benefit remains UNPROVEN; paired/target-only comparisons did not run.
