# DSpark fixed cache A/B harness

Run from the SGLang repository root using the service's Python environment.
This harness adds no inference code. It refuses an occupied port or an existing
output directory, and stops only its own managed process group.

## Protocol

- Exactly 128 / 512 / 2048 / 8192 input token IDs, crossed with concurrency 1 / 4 / 8.
- Reproducible natural-language document prompts with shared prefixes and distinct
  task suffixes. Dataset IDs, seed, server configuration, Git revision and package
  versions are saved. Same dataset and fixed output length across all modes.
- `full`: legacy DSpark guard forces complete prefill; sidecar inactive.
- `paired`: only draft-ready paired prefixes can be reused; incomplete draft
  readiness triggers the existing recovery plan. No guard relaxation.
- Optional `target_only_shadow`: paired sidecar observes but does not restrict
  target-only radix reuse. **Unsafe isolated control, not a deployable setting.**
  Requires `--include-target-only-shadow`. Plan counters in this mode are
  hypothetical recovery, not executed fallback.
- Fixed greedy sampling, PyTorch sampler, fixed seed. Framework deterministic
  inference stays OFF because FlashInfer deterministic mode disables radix cache.
  Greedy plus a seed is not a general determinism guarantee; the output gate is
  the evidence. All CUDA graphs OFF by default. `--enable-standard-graphs` permits
  existing prefill/draft graphs only; ragged target-verify graph always OFF.
- Each variant gets a cold serial reference with a cache flush before it. Each
  measurement repeat flushes and primes every variant before steady requests.
  Reference/prime/startup warmups are excluded from performance aggregates.
- Closed-loop concurrency; fixed request count divisible by 8, no random arrival
  rate. This is a cache-focused workload, not a production traffic trace.
- Every response (including cold and prime) must match the full-prefill reference
  token-by-token AND acceptance counts/histograms. Any discrepancy aborts the
  matrix, saves the mismatching response, and stops the managed service.
- CUDA/scheduler errors abort. With at least `--min-p95-samples` per cell/repeat
  (default 64), any paired-vs-full p95 regression in TTFT, TPOT or E2E aborts by
  default. Smaller smoke p95 values are descriptive, not a regression verdict.
  No automatic Git reset/checkout: inference code never changes and all existing
  untracked baseline files are preserved.

## Metrics and limitations

- TTFT: client request send to the first token-bearing SSE frame, not HTTP headers
  or the first metadata-only event.
- TPOT: `(last token receipt - first token receipt) / (output_tokens - 1)`.
  This is client-observed amortized TPOT. Speculative output can arrive in bursts;
  raw frame timestamps/token counts are saved, no fake individual token ITLs.
- p50/p95/p99: per-cell TTFT, TPOT and E2E, linear interpolation. Cold vs primed
  responses are preserved separately in `requests.jsonl`.
- GPU: device-wide nvidia-smi memory/utilization/temperature sampled every 100 ms.
  Not a PyTorch allocator peak or a guarantee of catching transient allocation.
  Use a dedicated GPU; background GPU consumers would contaminate this metric.
- Acceptance: total correct drafts / total proposed drafts; total completion
  tokens / total verify calls, plus full per-request counts and histograms.
- Request target-hit rate: fraction with `cached_tokens > 0`.
- Actual cached-token rate: sum cached tokens / sum `(prompt_tokens - 1)`.
- Shadow/paired candidate token hit rates: sidecar target/paired token counter
  deltas / eligible prompt tokens. If match calls differ from completed requests
  (e.g. scheduling retries), request-based token rates are null, not fabricated.
- Paired/target coverage and mean paired length per match are also saved.
- Partial/full recovery, recovered tokens and metadata evictions: sidecar plan
  counters. There is **no independently exposed actual draft-worker fallback
  counter**. Do not relabel `full_recovery_calls` as measured worker fallbacks.
  Zero events mean not exercised, not that partial recovery was validated.
- Dataset fits the normal sidecar budget in many cases. Budget pressure and
  eviction/reload/abort tests are separate experiments. `--cache-budget-fraction`
  can configure a separate run, but must not be mixed into the normal A/B claim.
- Fixed mode order: full, paired, optional target-only-shadow. Repeats characterize
  each cell; thermal/order bias is not eliminated. Independent counterbalanced
  runs and realistic traces are required before claiming stable-state benefit.

## Commands

CPU-only unit tests:

```bash
python benchmark/dspark/test_dspark_prefix_ab.py
```

Complete 12-cell smoke per mode: 8 steady requests, 1 repeat, 32 output tokens.
This is integration/correctness validation, not a p95 performance proof.

```bash
PYTHONPATH=python /root/conda-envs/sglang/bin/python benchmark/dspark/dspark_prefix_ab.py \
  --repo /root/autodl-tmp/src/sglang-main \
  --model /root/autodl-tmp/models/Qwen3-14B \
  --draft /root/autodl-tmp/models/dspark_qwen3_14b_block7 \
  --sps-table /root/autodl-tmp/results/sps-probes/dspark14-sps-probe-20260830-210115/qwen3-14b-rtxpro6000-additive-sps.json \
  --preset smoke --include-target-only-shadow \
  --output /root/autodl-tmp/results/dspark-cache-ab-SMOKE-UNIQUE-RUN-ID
```

Standard: 64 steady requests x 3 repeats per cell/mode, 128 output tokens.

```bash
PYTHONPATH=python /root/conda-envs/sglang/bin/python benchmark/dspark/dspark_prefix_ab.py \
  --model /root/autodl-tmp/models/Qwen3-14B \
  --draft /root/autodl-tmp/models/dspark_qwen3_14b_block7 \
  --preset standard --include-target-only-shadow \
  --output /root/autodl-tmp/results/dspark-cache-ab-STANDARD-UNIQUE-RUN-ID
```

Supply the same SPS table for all compared runs, if used. Output directory must
not exist. Harness returns 0 only for a completed matrix, 2 for a safety stop.
Interrupted/safety-stopped runs remain incomplete and cannot establish benefit.

Artifacts: `manifest.json`, exact `dataset.json`, per-mode `launch.json`,
`server_info.json`, `server.log`, `gpu_samples.jsonl`, raw `requests.jsonl`,
`shutdown.json`; top-level `cells.jsonl`, `summary.csv`, `report.md`, `status.json`.
No existing files or result directories are deleted.
