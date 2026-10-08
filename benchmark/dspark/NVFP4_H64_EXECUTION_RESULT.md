# h64 real-task execution result: stopped, NOT a quality/benefit result

Evidence root: `/root/autodl-tmp/dspark-quant-quality-20261009-h64`.
Immutable dataset SHA256:
`c4ba785e66d5535c7d454a25c19af692aab686c0ea94df204a12b0f6a649488f`.

Both BF16 ordinary-AR and BF16 DSpark real-task sentinel gates passed all 56
requests apiece, including variable EOS, long reasoning and c1/c4/c8 repeats.
DSpark tokens matched its BF16 AR references, and all five acceptance fields
matched repeat/concurrent runs. This is bounded execution evidence, not quality.

The NVFP4 CUTLASS ordinary-AR gate completed three requests. Its fourth request
(MATH intermediate_algebra/345, output cap 4096) exceeded the fixed 240-second
wall deadline. The service was stopped; quantized DSpark sentinels, 384-item
quality and stable A/B did not start. Preserve the failed route and its timeout;
do not retry it by raising the deadline or removing the problem.

Before the stop the scheduler continued generating at approximately 14 tokens/s.
The completed long MATH output had 2318 tokens and TPOT 72.109 ms. The BF16 AR
counterpart had 2392 tokens and TPOT 51.85 ms. These heterogeneous, unconditioned
AR timings are diagnostic only: they do not measure DSpark steady E2E benefit,
nor imply an illegal access, numerical mismatch or a stalled GPU.

All owned services stopped and the serving checkout retained HEAD
`0577109f4e659b8ef2ef3f979f1fd7f079b25821`; main and all original untracked files
were preserved. No dependency upgrade or inference-path edit was made.

Next route: separately attest/profile the already-installed cuDNN FP4 backend,
with a new within-backend AR oracle. Same checkpoint, BF16 head/draft/KV, fixed
FA4 policy, fixed quality data/grader and human <=1 percentage point task budget.
Do not mix CUTLASS and cuDNN references or treat backend smoke as model quality.

Research lead (not an established local cause):
[FlashInfer issue 4549](https://github.com/flashinfer-ai/flashinfer/issues/4549)
reports small-M SM120 workspace allocation in an older FlashInfer version.
Verify the installed implementation and bounded nsys trace before attributing
the local slowdown to that issue.
