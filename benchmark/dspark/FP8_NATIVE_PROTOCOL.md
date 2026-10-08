# h66 independent W8A8 FP8 alternative

Branch `exp/quant-w8a8-fp8-native-target-bf16-draft`, not main or serving checkout.
Decision comes from retained FP4 decode deadline failure and bounded nsys host
dispatch analysis; precision/backend alternatives are not mixed conclusions.

Use the already-installed SGLang `--quantization w8a8_fp8`, loading the unchanged
original BF16 parent. Existing loader performs static per-output-channel FP8
weight quantization and dynamic per-token FP8 activation quantization. This
requires neither a new model download nor ModelOpt/dependency upgrade.

Native `sgl_kernel.gemm.fp8_scaled_mm` CUTLASS path only:
`USE_TRITON_W8A8_FP8_KERNEL=0`, `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0`. Reject
unsupported hardware, per-tensor weight fallback, quantized head/draft or an
unexpected method. No handwritten replacement GEMM. Do not set a blockwise-only
FP8 backend flag and mislabel this per-channel path as controlled by that flag.

Attest 160 `W8A8Fp8LinearMethod` weights in FP8 E4M3, transposed [K,N], FP32
[1,N] channel scales, dynamic input scale, native dispatch enabled and all
dimensions divisible by 16. BF16 shared head/embedding, 20 draft linears and
target/draft KV unchanged. Same fixed FA4 policy; graph/cache/adaptive/compact
features OFF. Recipe implementation source hashes are frozen for the run.

First native backend smoke covers four actual Qwen sizes and M1/8/32/64. Then
new FP8 AR tokens, exact DSpark agreement and c1/c4/c8 repeat/acceptance gates,
including variable EOS/long reasoning, with no weakened timeout/guards. Reuse
the unchanged passed BF16 sentinel evidence read-only, not as new quality data.

Same 384 heldout items, exact input IDs/caps, isolated grader and <=1 percentage
point per-task score-loss budget. Fixed subset scores and paired uncertainty are
not universal statistical noninferiority. Only acceptable quality launches the
unprofiled equal16K-KV control+ABBA: >1% mean E2E gain in both orders, zero p95
TTFT/TPOT/E2E regression. No candidate timing after failed execution/quality.

Research: [SGLang quantization documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantization.mdx)
and actual installed `w8a8_fp8.py`/`fp8_utils.py` dispatch (the latter supports
Triton tuning by default, explicitly disabled here). No other-model benchmark
or BF16-vs-FP8 token difference is counted as success/failure of execution.

Initial standalone smoke failed before its first native quantization call:
Ninja 1.13 was installed in the serving environment but not in the standalone
shell PATH. Match ManagedServer's existing executable-parent PATH convention;
no dependency change. Preserve h66 failed setup evidence and run into a fresh
`dspark-quant-fp8-quality-20261009-h66-v2` directory. No numerical/safety failure
or scored item is retried.
