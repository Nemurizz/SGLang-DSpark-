# Independent NVFP4 target experiment — 2026-10-08 / h63

## Decision and scope

Branch `exp/quant-nvfp4-target-bf16-draft` starts at `0577109` (qualified stable
FA4 fixed64 numerical baseline). The serving checkout stays detached at that
baseline; the branch contains tools/configuration only and imports its source
through an explicit `PYTHONPATH`. This is necessary because the original checkout
contains 8,357 preserved upstream/untracked files absent from a plain worktree.
Do not silently infer that the tracked branch is a complete serving installation.

Do not modify main, old BF16 references/weights/packages, or the earlier graph,
cache, residual-norm and stall-diagnostic branches. Never load h62 `sitecustomize`
or asynchronous faulthandler instrumentation. Only owned test services may stop.

First recipe:

- Target: public NVIDIA Qwen3-14B NVFP4, **W4A4** transformer linears, not W4A16.
- Revision: `bc39319a4dc265d9bbb9a9731bc52c4988d9ece7`, ModelOpt producer 0.35.0.
- BF16 LM head and embedding, BF16 unquantized DSpark draft, BF16 target/draft KV.
- Explicit `--quantization modelopt_fp4`, existing FlashInfer 0.6.17 **native
  SM120 CUTLASS** FP4 backend; no new GEMM kernel.
- FA4 fixed64 attention, splits=1, deterministic settings for BF16 operations.
- CUDA graphs/radix/paired cache/adaptive draft/compact CUDA OFF.
- Existing SPS table unchanged. Quantization may lower acceptance, so acceptance
  is an outcome, not assumed to match BF16.

Quantized custom GEMMs are not made batch invariant merely by turning on the
BF16 deterministic flag. A new target-AR oracle and exact within-configuration
token/acceptance gates are mandatory. Quantization is NOT a fix for the prior
unexplained ~60s baseline stall.

## Research before GPU work

1. [SGLang official quantization documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantization.mdx):
   offline checkpoint route and native SM120 FlashInfer CUTLASS. Online
   `nvfp4_online` is MoE-only SM100/103; Qwen3-14B dense/SM120 must not use it.
   Installed source, not only current documentation, was inspected.
2. [NVIDIA same-model checkpoint](https://huggingface.co/nvidia/Qwen3-14B-NVFP4)
   and [quantization metadata](https://huggingface.co/nvidia/Qwen3-14B-NVFP4/raw/main/hf_quant_config.json):
   only transformer linears W4A4; LM head excluded. It was calibrated using
   CNN/DailyMail and is advertised for TensorRT-LLM/B200. This does **not** prove
   quality, compatibility or performance in our SGLang/DSpark/SM120 workload.
   Embedded config asks for FP8 KV; explicitly override with BF16 and attest
   actual KV storage before requests count as passed.
3. [DSpark packed LM-head issue #35354](https://github.com/sgl-project/sglang/issues/35354):
   raw matmul of packed weights has a logical/physical shape mismatch. The
   installed DSpark source already has quant-method dispatch; additionally our
   chosen head stays BF16. Do not reduce memory or loosen graph guards to fix
   a tensor layout problem.
4. [Community PR #34966](https://github.com/sgl-project/sglang/pull/34966):
   examines packed-head dispatch and BF16 tiny-scale problems, but its Marlin
   SM80/90/RTX4090 route is not our native SM120 route. It is open, not a validated
   drop-in patch for this project; no cherry-pick.
5. [NVFP4 KV/speculative issue #36010](https://github.com/sgl-project/sglang/issues/36010):
   another reason to keep KV BF16 initially. Weight quantization and KV
   quantization must not be conflated.
6. [NVIDIA Local-Hessian blog, 2026-09-09](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html):
   calibration can improve scale selection without changing runtime format.
   Its Qwen3.5/Qwen3.8 results are not evidence for our old Qwen3-14B checkpoint.
   Keep this as an accuracy-recovery fallback only if the existing checkpoint
   fails a real quality gate; do not install a fresh calibration stack first.
7. [Author's SM120 Qwen3.8 deployment](https://github.com/darksidewalker/qwen3.8-27b-sglang-dspark-blackwell):
   integration inspiration only. Different model, head/KV recipe, draft and graph
   settings prevent importing the reported speed as our A/B result.
8. [FlashInfer issue #3398](https://github.com/flashinfer-ai/flashinfer/issues/3398)
   initially reported M-dependent zero outputs. Its merged
   [fix #3497](https://github.com/flashinfer-ai/flashinfer/pull/3497) identified
   incorrect global-scale dtype interpretation, not a proven dispatch problem.
   The installed 0.6.17 wrapper normalizes scales to FP32. Attest actual target
   `alpha` and `input_scale_inv` as FP32 scalars on every fused linear, packed
   U8 weights, FP8 block scales and BF16 output; do not copy a speculative root
   cause from an issue description into our diagnosis.
9. [SGLang issue #36291](https://github.com/sgl-project/sglang/issues/36291)
   reports a deterministic-flag coverage gap on a different Qwen3.8 hybrid
   Mamba/DFlash2/SM121 configuration. It does not establish our dense Qwen3
   failure, but reinforces using actual output tokens instead of configuration
   or task scores alone as the reproducibility test.

## Why this is worth one bounded test

This avoids both a new kernel implementation and dependency churn. The native
backend is installed; published same-parent weights exist; BF16 shared head
avoids a known DSpark integration trap. Theoretical benefits are reduced linear
weight traffic and native FP4 tensor-core compute. Countervailing costs are
activation quantization, small-M kernel overhead and lower draft acceptance.
No theoretical bandwidth ratio is an end-to-end speedup claim.

## Execution and stop gates

1. Native backend smoke: actual Qwen3-14B fused linear dimensions (QKV/o/gate-up/
   down), M=1/8/32/64, finite outputs and repeated-row exact equality. This is
   availability only, not model-level correctness or performance.
2. Download ~10.544 GB of pinned weights to the separate shared volume (local
   XFS initially had only ~2GB free); cross-check the three LFS SHA256 values against the
   official HF API read from the desktop. No overwrite or cleanup of baseline.
   Initial network attempts: official HF unreachable remotely, hf-mirror works.
   On 2026-10-09 the user expanded local XFS to 80GB. Preserve the shared source
   and old models; `stage_verified_checkpoint.py` copies only pinned manifest
   entries to a fresh local folder, hashes every copy and publishes its verified
   manifest last. Never time inference loading directly from shared FUSE.
3. Audit same architecture and bitwise-identical BF16 embedding/LM-head tensor
   bytes against the preserved original checkpoint; no model swap disguised as
   quantization. New quantized ordinary target, graph/cache OFF, eight fixed512 prompts,
   output128; serial and c4/c8 repeatability. Freeze a **separate** quant oracle.
4. Same quantized target + BF16 DSpark: token-by-token equality to that oracle,
   then repeat serial/c4/c8 with all five acceptance fields exact within DSpark.
   On any mismatch/error, stop, preserve logs, keep serving baseline unchanged;
   do not rewrite verify/planner or loosen checks. Synthetic prompts here are
   execution tests only, not a quality benchmark.
5. Before a success claim: independent, representative language/code/math
   quality evaluation against BF16, including long reasoning. User has not
   specified an acceptable quality-loss budget; report measured loss and obtain
   a quality acceptance decision before promoting a lossy model configuration.
6. Only after correctness/quality: one p2048/output128/c1 stable A-B-B-A test,
   matched inputs/output lengths and runtime settings; save TTFT, TPOT, E2E,
   p95, acceptance and device memory. Quantized content may differ, so never
   label a BF16-vs-NVFP4 token mismatch itself a quant correctness failure.
   To claim >1%: both AB/BA directions >1%, baseline-BB noise checked, no p95
   regression; then expand workload. This baseline has strict BF16 numerical
   overhead and is not a general claim against fastest production BF16.

Hot serving logs and request JSONL belong on local XFS, not shared FUSE. Startup
weight loading and download are not steady-state timing. Keep fixed KV capacity
for fair memory accounting where possible: a larger automatic KV pool can hide
the reduction in model memory. Cache metrics are disabled/not applicable here,
not paired-cache wins.

## Current result

The native backend smoke passed all 16 shape cases. Twelve harness/parent/staging
unit tests passed locally and remotely. Download and local staging hashes passed,
architecture matched, and BF16 embedding/LM-head tensor bytes matched the preserved
original checkpoint. Installed
FlashInfer `fp4_gemm_template_sm120.h` declares FP32 `ElementAccumulator` and
`ElementCompute`; BF16 output still introduces rounding. No model quality,
general DSpark correctness or end-to-end gain has yet been proved.
`NVFP4_TARGET_BOUNDED_RESULT.md` records the completed bounded AR/DSpark gates:
112 requests passed on fixed512/output128/c1/c4/c8. This does not cover quality
or stabilized performance.

The first target launch (target-gate-01) stopped before any completed test request:
offline auto-detection loaded `modelopt_fp4`, but `/server_info` retained
`quantization=None`. Explicitly pinning `--quantization modelopt_fp4` makes the
same checkpoint format auditable without weakening the metadata check. All
first-launch evidence is preserved; v2 uses a fresh output folder. This startup
attestation failure is not an output mismatch or measured numerical failure.
