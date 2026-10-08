# NVFP4 target / BF16 draft: bounded feasibility passed, 2026-10-09

## Result and limits

The independent branch `exp/quant-nvfp4-target-bf16-draft` passes its first
real-model execution gate. This is **not** an end-to-end speedup or model-quality
claim, and does not qualify high concurrency above 8, long reasoning, 8K prompts,
graphs, prefix caching or FP4/FP8 KV.

Serving source stayed at `0577109f4e659b8ef2ef3f979f1fd7f079b25821`; main stayed at
`0c1f1eb64af939379ce47045c56aa8d2508c54df`. Both phase manifests recorded identical
before/after serving-source identities. The 8,357 original untracked paths, old
BF16 oracle and dataset were preserved. No serving-source edit, reset, dependency
upgrade, force push or GitHub push was performed for this experiment.

## What actually passed

- Native FP4 backend availability smoke: 16 Qwen3 fused-linear shape cases.
  This is not a correctness certificate for the whole model or a speed benchmark.
- Twelve tool unit tests passed locally and remotely.
- Public NVIDIA Qwen3-14B NVFP4 revision
  `bc39319a4dc265d9bbb9a9731bc52c4988d9ece7`: all download and local-copy hashes
  passed, including three independently origin-checked LFS SHA256 digests.
- Same architecture; raw BF16 embedding and LM-head bytes exactly match the
  original BF16 checkpoint. Transformer weights are packed NVFP4.
- Actual target storage: 160 fused `ModelOptFp4LinearMethod` linears, U8 weights,
  FP8 block scales, FP32 scalar `alpha`/`input_scale_inv`, BF16 output,
  FlashInfer CUTLASS backend. Actual draft storage: 20 BF16 unquantized linears.
- Actual target/draft KV BF16, fixed64 FA4 / split1, deterministic BF16 policy;
  all CUDA graphs, radix/paired cache, adaptive draft and CUDA compact OFF.

| Phase | Requests | Token checks | Repeatability |
| --- | ---: | --- | --- |
| Quantized target AR | 56 | 8 new references + 48 exact comparisons | c1/c4/c8, two repeats each |
| Same quantized target + BF16 DSpark | 56 | 8 exact comparisons to quantized AR + 48 exact DSpark repeats | c1/c4/c8, all five acceptance fields exact |

Every request used a fixed 512-token input and exactly 128 output tokens, greedy
sampling, zero cached tokens and zero retractions. The eight synthetic prompts
are distinct; this gate is an execution workload, **not** a quality benchmark.
The quantized oracle is separate from the immutable BF16 oracle. No equality to
old BF16 output or acceptance was required for the new quantized math contract.

The eight DSpark reference requests total 295 verifies, 742 accepted drafts and
2,065 proposed drafts: acceptance 35.9322%, completion tokens per verify 3.47119.
These are descriptive results for this workload, not a BF16-vs-quant speed claim.

## Startup and hardware notes

The user expanded local XFS from 50GB to 80GB. GPU/driver remained the same:
RTX PRO 6000 Blackwell Server Edition / SM120 / 595.71.05, UUID
`GPU-7361f058-4f4f-325a-0075-2b03e0838996`.

The first target launch was safety-stopped with **zero completed test requests**:
auto-detection loaded ModelOpt FP4, but `/server_info` still reported
`quantization=None`. The v2 profile explicitly pins `--quantization modelopt_fp4`;
metadata checks were not loosened. Both launches and all logs are retained.
Shutdown SIGTERM/SIGQUIT cleanup messages are not a newly observed inference crash.

Sampled device-memory peaks were 55,047 MiB (AR) and 55,193 MiB (DSpark), including
startup. Automatic KV capacity differs between configurations, so these peaks
are **not** a fair BF16-vs-FP4 memory-savings comparison. TTFT/TPOT/E2E and stream
events are saved per request, but no stabilized A/B or sufficiently sampled p95
performance gate has been run. Owned services stopped; GPU returned to 0 MiB.

## Evidence

Remote evidence root: `/root/autodl-tmp/dspark-quant-feasibility-20261008-h63`.

- `backend_smoke.json`, `parent_audit.json`, `stage.log`.
- `target-gate-01`: preserved startup attestation rejection.
- `target-gate-02`: passed target gate, manifest, new references, requests,
  launch/configuration, runtime storage/attention attestations, GPU samples.
- `dspark-gate-01`: passed DSpark gate with the same artifacts and draft attestation.

Pinned local checkpoint root:
`/root/autodl-tmp/models/Qwen3-14B-NVFP4-20261008-h63`.
Its shared-volume source and the original BF16/draft checkpoints remain untouched.

## Next decision, not yet executed

First measure representative language/code/math quality against BF16, including
long reasoning; do not treat identical synthetic output as evidence of retained
task accuracy. Report any loss before choosing an acceptable quality budget.
Then run one matched p2048/output128/c1 steady-state A-B-B-A comparison, preserving
the strict baseline and independent quant oracle. Require >1% E2E improvement in
both order directions, stable BF16 control windows, and no TTFT/TPOT/E2E p95
regression before calling the optimization successful. Do not expand the matrix
or introduce another kernel/backend until this evidence justifies it.

Research rationale and primary-source links are in `NVFP4_TARGET_FEASIBILITY.md`.
