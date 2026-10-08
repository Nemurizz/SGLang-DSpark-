# h64 real-task quality protocol (fixed before scoring)

This independent quantization experiment preserves main, old numerical baseline,
untracked upstream files, old reference JSONL and h63 feasibility tag. No graph,
cache, draft-length or KV-precision change is combined with quantization.

## Immutable evaluation specification

- Seed/ranking namespace: `dspark-quality-20261009-h64-v1`. SHA256 ranking by
  task and stable item ID; selection sees no model output.
- GSM8K official test: 128/1319, zero-shot, thinking off, 1024 token cap.
- ARC-Challenge official test: 128/1172, generative letters, thinking off,
  256 token cap. This is NOT the standard log-likelihood ARC leaderboard score.
- Sanitized MBPP: 64/257 eligible tasks with official test IDs 11..510;
  provided public assertions are included in the prompt, thinking off,
  1024 token cap. Functional pass@1 using all provided tests, not EvalPlus.
- MATH-500: 64/351 eligible pure-text scalar rational-answer problems, thinking
  on, 4096 token cap. Eligibility is gold-only and excludes Asymptote diagrams.
  This is NOT a full MATH-500 score or symbolic-expression evaluation.
- Both recipes use the identical preserved tokenizer/chat template, greedy
  decoding, input IDs and output caps. EOS enabled. Truncated outputs are wrong.
- Strict final-marker/boxed rational grading; no result-dependent extraction
  changes, item replacement, request retries or outlier removal.

Sources are pinned in `quality_data.py`. Downloads from official GitHub/HF were
performed on the desktop because the GPU instance's raw GitHub request timed
out; original TLS bytes were then staged by SSH. SHA256:

| Source | SHA256 |
| --- | --- |
| GSM8K test | 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14 |
| Sanitized MBPP | ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9 |
| MATH-500 test | 35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132 |
| ARC-Challenge test parquet | 62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9 |

Generated dataset SHA256:
`c4ba785e66d5535c7d454a25c19af692aab686c0ea94df204a12b0f6a649488f`.

## Grader safety / execution prerequisites

Generated Python executes as UID/GID 65534 inside a readonly chroot with copied
stdlib/libraries, no host mounts or credentials, only a readonly urandom device.
`no_new_privs`, seccomp denies sockets, processes/exec, host namespace/mount and
privileged operations. CPU 2 seconds, address space 256 MiB, wall 4 seconds,
stdout 64 KiB, filesystem writes denied. No generated code executes as root.
Canonical MBPP solutions must all pass before model scoring (64/64 passed).
Benign math code and four isolation probes passed; six grader unit tests passed.

For each math contract independently, eight real-task sentinels (first two
fixed selected items per category) are compared serially and at c1/c4/c8,
two repeats. First ordinary AR tokens establish the within-contract oracle,
then DSpark must match it exactly; DSpark repeats require all five acceptance
fields to match. Variable EOS and long reasoning are included. Any mismatch,
illegal memory access or server exception stops that service/route immediately.
Cross-precision token differences are expected and scored, not misclassified as
execution bugs.

The 384-item quality run follows only successful execution gates, concurrency 8.
Report correct totals, per-item paired discordances, truncations and uncertainty.
Initial acceptance policy: no observed per-category score loss against BF16.
Human subsequently explicitly authorized **at most 1 percentage point loss per
task**, before paired quality scores were available (2026-10-09). Only the
acceptance rule changes; the dataset SHA, prompts, caps, selection and grader
remain identical. The report implements the budget with exact integer counts:
`100 * (bf16_correct - candidate_correct) <= n`. Thus a one-item loss out of 128
passes (0.78125 points), but one out of 64 fails (1.5625 points). This is an
observed subset criterion, not a claim of statistical noninferiority everywhere.

## Performance prerequisite and alternatives

Heterogeneous quality-request timings are NOT a steady-state speed comparison.
After acceptable quality, run unprofiled fixed p2048/c1/output128 BF16 controls
and A-B-B-A with equal warmup/conditioning, 64 requests/window, original noise
admission and zero p95-regression tolerance. Both orders must improve mean
end-to-end latency by >1%. Freeze separate exact within-recipe token/acceptance
oracles; preserve all observations and device telemetry.

If quality loses: investigate errors/truncations first; prefer conservative
MLP-only NVFP4/BF16 attention or FP8 rather than arbitrary loss tolerance.
If performance loses: profile measured work and consider an independently
qualified installed cuDNN backend; never infer E2E benefit from GEMM timings.

Primary research informing alternatives:

- [SGLang quantization compatibility/backends](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantization.mdx): SM120 native CUTLASS,
  cuDNN CUDA13+/9.15+, mixed fused-layer compatibility caveat.
- [NVIDIA MLP-only recipe](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/presets/model/nvfp4_mlp_only.yaml): narrower precision scope.
- [NVIDIA Local-Hessian](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html): offline scale calibration can recover accuracy without changing
  runtime format; examples are different Qwen models, NOT proof for Qwen3-14B.
- [SGLang issue 36291](https://github.com/sgl-project/sglang/issues/36291): deterministic flag alone does not certify custom quantized GEMM batch invariance.

Status: quality and E2E benefit not yet proven when this protocol was committed.
