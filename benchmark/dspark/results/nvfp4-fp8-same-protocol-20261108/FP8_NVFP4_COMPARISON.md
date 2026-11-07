# FP8 versus NVFP4 under the same DSpark paired-prefix protocol

FP8 source: h74; NVFP4 source: h77. Both use 8 cells, 32 measured requests/cell, 2 warmup batches, BF16 draft, DSpark block7, FA4 fixed64, Graph/compact-scan/adaptive-draft/KV quantization off, paired prefix cache, max_total_tokens=32768, output=128, and AB/BA counterbalancing.

| cell | FP8 E2E gain | NVFP4 E2E gain | FP8 TTFT p95 ratio | NVFP4 TTFT p95 ratio | FP8 TPOT p95 ratio | NVFP4 TPOT p95 ratio | BF16 peak | FP8 peak | NVFP4 peak | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1024_c16 | 8.61% | -50.71% | 0.863 | 1.530 | 0.924 | 1.271 | 41801 | 29733 | 24043 |
| p1024_c4 | 6.30% | -79.96% | 0.834 | 1.801 | 0.838 | 1.857 | 41257 | 29189 | 23227 |
| p1024_c8 | 5.37% | -62.05% | 0.972 | 1.609 | 1.218 | 1.677 | 41257 | 29189 | 23227 |
| p2048_c4 | 2.76% | -88.98% | 0.891 | 1.913 | 0.905 | 1.975 | 41801 | 29733 | 24043 |
| p2048_c8 | 2.13% | -85.15% | 0.860 | 1.776 | 1.067 | 1.727 | 41801 | 29733 | 24043 |
| p512_c16 | 4.97% | -60.45% | 0.995 | 1.610 | 0.942 | 1.600 | 41257 | 29189 | 23226 |
| p512_c4 | 7.13% | -70.48% | 0.923 | 1.599 | 0.942 | 1.632 | 39261 | 27193 | 21447 |
| p512_c8 | 3.67% | -61.06% | 0.934 | 1.375 | 1.042 | 1.746 | 39725 | 27657 | 21796 |

## Aggregate observations

- FP8 mean E2E gain over the 8 cells: 5.12%; NVFP4: -69.85%.
- FP8 mean TTFT p95 ratio: 0.909; NVFP4: 1.652.
- FP8 mean TPOT p95 ratio: 0.985; NVFP4: 1.685.
- Mean quantized peak memory: FP8 28952 MiB; NVFP4 23132 MiB.
- Mean acceptance: BF16 0.356; FP8 0.354; NVFP4 0.327.

NVFP4 has lower device peak memory but is slower in every paired cell under this DSpark protocol. The cause is a path-level mismatch: the ModelOpt NVFP4 target uses FP4 block-scale GEMM and quantized target execution, while DSpark fixed FA4 verify is a small, irregular per-step workload; activation/scale/layout overhead and lower effective acceptance outweigh the weight bandwidth savings. FP8 uses the native W8A8 scaled GEMM path and is better aligned with this DSpark workload.

