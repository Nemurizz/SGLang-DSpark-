# Independent cuDNN FP4 route, h65

Branch `exp/quant-nvfp4-cudnn-target-bf16-draft` starts from the retained h64
failure-record commit. No serving checkout, main, reference or dependency edit.
The installed FlashInfer 0.6.17 accepts `backend="cudnn"`; SGLang maps explicit
`--fp4-gemm-backend flashinfer_cudnn` to it. cuDNN 9.20/frontend 1.27 were already
installed. No third-party GEMM replacement is written.

Same official pinned Qwen3-14B NVFP4 checkpoint, same BF16 head/embedding,
unquantized BF16 draft and BF16 target/draft KV. FA4 fixed64 split1 and all
graphs/radix/paired cache/adaptive draft/compact CUDA OFF, as before. Only FP4
backend changes. cuDNN tokens are not compared with CUTLASS/BF16 tokens.

Backend smoke checks actual four Qwen linear dimensions at M1/8/32/64 (all 16
passed finite output and exact duplicated-row comparisons). This is availability
evidence, not a model quality or end-to-end certificate. The original smoke JSON
retains a legacy CUTLASS-only docstring; its explicit `backend` is `cudnn`.

The preserved BF16 real-task sentinel certificates are reused read-only after
checking dataset/source identities and 56 successful requests per arm. They
are not fabricated, copied as fresh runs, or reclassified as quality scores.
New cuDNN ordinary AR and DSpark real-task gates cover serial/c1/c4/c8, two
repeats, variable EOS and long MATH reasoning. Existing wall time limits remain.
Separate backend-labelled references and storage/server-info validation prevent
mixing numerical contracts. CUTLASS remains the strict default in old helpers.

Initial suite preflight refused a documentation/transcription hash with an
extra digit, before launching any model. The file and both existing BF16 gate
manifests agreed on the actual 64-character SHA256
`c4ba785e66d5535c7d454a25c19af692aab686c0ea94df204a12b0f6a649488f`.
Correct the recorded hash only; preserve initial suite.log and launch into a
separate suite-v2.log. No evaluated request/input or failed safety gate is retried.

Only then generate the same 384 scored BF16/candidate DSpark items with the same
immutable dataset and jailed grader. Human criterion: each task score loss at
most 1 percentage point; integer comparisons and paired uncertainty unchanged.
Only acceptable quality launches the already-fixed unprofiled equal-16K-KV
capacity p2048/c1/output128 control + ABBA protocol. Each order requires >1%
mean E2E improvement and no p95 TTFT/TPOT/E2E regression. No speed conclusion
from quality timings, backend smoke or profiler.

The separate 64-token ordinary target CUDA/NVTX diagnostic reuses installed
native calls without modifying operands/tactics/scales. Profiler 2025.6.3 uses
the previously audited CUDA13.2 software collector and exports SQLite plus
coverage diagnostics. Diagnostic hooks never enter scored/timed services.

Research: [FlashInfer FP4 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html),
[SGLang backend docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantization.mdx),
[small-M workspace report](https://github.com/flashinfer-ai/flashinfer/issues/4549).
The community issue is a hypothesis until locally measured; different model,
version or hardware results are not our E2E proof.
