# Independent numerical baseline: eager deterministic FA4

The user explicitly authorized this independent arithmetic contract while
preserving ALL old references. This is correctness qualification, not an
optimization or performance result. New-profile output need not reproduce old
BF16 cuBLAS rounding, but ordinary target greedy output, DSpark serial/concurrent
output and the five DSpark acceptance fields remain strict gates.

No inference-source, kernel, graph, quantization, or main-branch changes.
The explicit harness uses existing framework batch-invariant operations and
FA4 for target prefill/verify/decode AND draft attention, BF16 target/KV, TP1,
page size 1, all graphs OFF. No batch-variant mm fallback. The rejected low-
precision-reduction probe is OFF. This does not bypass FlashInfer's cache guard:
FA4 is already listed in the framework's radix-supported deterministic backends.

The profile digest records core source SHA256 and package versions and salts
the paired sidecar fingerprint. A read-only worker hook attests batch-invariant
Torch/FA4 flags, backend implementation, actual KV dtype and fixed split counts.
The hook does not clone or modify numerical tensors and is NOT timing evidence.

Gate order (one owned service at a time, stop on first failure):

1. Independent ordinary target: cold serial reference, cold repeat, then three
   serial/concurrent rounds; prefix reuse disabled.
2. DSpark full prefill: match independent target token-by-token; then cold repeat
   and three serial/concurrent rounds match all five acceptance fields exactly.
3. Optional paired arm ONLY after both prior arms pass: cold control, distinct-
   suffix priming, then three warm serial/concurrent rounds. Actual cached_tokens
   must be positive in warm checks, otherwise no cache qualification.

Start with `128 x 1/4/8`, 32 output tokens, then the full fixed 12-cell matrix and
longer output. No complete reusable baseline oracle is published from a focused
subset. Partial-recovery counters are observations of plans, NOT proof of real
worker recovery; targeted partial tests and instrumentation still remain.

Example (new output directory required):

```sh
PYTHONPATH=python:benchmark/dspark python benchmark/dspark/stable_baseline_gate.py \
  --template /root/autodl-tmp/results/dspark-cache-compatible-original-baseline-20261007-h21 \
  --expected-head COMMIT_SHA --output NEW_OUTPUT_DIR
```

Only after correctness qualification may a separate counterbalanced A/B harness
claim any paired-cache benefit. Same numerical profile in both arms, independent
frozen controls, real cache hits, correct draft recovery, sufficient repeated
samples and TTFT/TPOT/p95/acceptance/memory gates are all required. Baseline
configuration cost is not paired-cache benefit.

## v1 rejected; explicitly isolated v2 candidate

h26: ordinary target `128 x 1/4/8`, three serial/concurrent rounds passed.
The FIRST DSpark cold B1 request differed at output index 2 (34334 vs16555).
Services stopped and checkout rolled back to a61; no paired run or oracle was
published. Tag `dspark-stable-fa4-target-gate-failed-20261007` preserves v1.

h27 read-only committed-anchor traces: QKV inputs/outputs, Q/K/V, past KV and
KV writes bitwise identical at position128/token576. First observed difference
is layer0 attention (113/5120 BF16 elements, max0.0001220703125). Two repeats
per arm are bitwise stable. Saved-Q/K/V replay reproduces BOTH observed outputs
exactly. Automatic host tile is decode64x64 vs verify64x128. Existing fixed
64x64 AND128x64 APIs yield bitwise-equal anchor outputs for Q1/Q8 at this saved
boundary. That is numerical causality evidence, not a full-model correctness
pass, not proof for other shapes, and not performance evidence.

`--fixed-fa4-tile` selects an independent v2 digest and explicit opt-in factory.
The factory mutates ONLY the process-local SM120 host selector, before any FA4
compile, to existing64x64 kernels. It is NOT a read-only observer. It is not
loaded by ordinary/default service startup. No production or untracked source
file is edited; worker exit removes the process-local policy. Graphs, local
attention, bias and non-Qwen3 geometry fail closed. Fixed tile applies equally
to target prefill/decode/verify AND draft in that worker. Runtime attestation
requires applied selector calls. All original correctness gates still apply.

No new GEMM, attention kernel, graph or quantization experiment is implemented.
New profile/source/package/model-metadata/SPS/hardware digests isolate refs and
paired sidecar fingerprints. v2 must be independently qualified before use.
