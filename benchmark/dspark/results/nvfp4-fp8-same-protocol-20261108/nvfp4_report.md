# NVFP4 high-concurrency DSpark A/B

Status: completed

Protocol: standard_cuda_graphs=False; ragged DSpark graph/compact scan/adaptive draft/KV quantization OFF; cache_mode=paired; max_total_tokens=32768; output=128; 32 measured requests per cell; 2 warmup batches; BF16→NVFP4→NVFP4→BF16.
Correctness is exact within each recipe (tokens plus five DSpark acceptance fields). BF16/NVFP4 cross-recipe token equality is not required.

| Cell | Order | E2E gain mean | E2E gain p95 | TTFT p95 ratio | TPOT p95 ratio | BF16/NVFP4 peak MiB | BF16/NVFP4 acceptance | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| p1024_c16 | AB | -51.33% | -40.76% | 1.537 | 1.272 | 41801 / 24043 | 0.316 / 0.340 | FAIL |
| p1024_c16 | BA | -50.08% | -39.72% | 1.524 | 1.270 | 41801 / 24043 | 0.316 / 0.340 | FAIL |
| p1024_c4 | AB | -78.43% | -81.82% | 1.785 | 1.826 | 41257 / 23227 | 0.371 / 0.315 | FAIL |
| p1024_c4 | BA | -81.49% | -85.36% | 1.817 | 1.887 | 41257 / 23227 | 0.371 / 0.315 | FAIL |
| p1024_c8 | AB | -59.97% | -57.91% | 1.590 | 1.661 | 41257 / 23227 | 0.356 / 0.332 | FAIL |
| p1024_c8 | BA | -64.13% | -61.96% | 1.629 | 1.692 | 41257 / 23227 | 0.356 / 0.332 | FAIL |
| p2048_c4 | AB | -88.27% | -92.97% | 1.910 | 1.970 | 41801 / 24043 | 0.385 / 0.308 | FAIL |
| p2048_c4 | BA | -89.68% | -94.50% | 1.917 | 1.979 | 41801 / 24043 | 0.385 / 0.308 | FAIL |
| p2048_c8 | AB | -86.65% | -82.06% | 1.789 | 1.738 | 41801 / 24043 | 0.384 / 0.306 | FAIL |
| p2048_c8 | BA | -83.65% | -79.76% | 1.763 | 1.717 | 41801 / 24043 | 0.384 / 0.306 | FAIL |
| p512_c16 | AB | -59.62% | -50.93% | 1.605 | 1.593 | 41257 / 23227 | 0.346 / 0.352 | FAIL |
| p512_c16 | BA | -61.28% | -52.24% | 1.616 | 1.606 | 41257 / 23225 | 0.346 / 0.352 | FAIL |
| p512_c4 | AB | -70.40% | -62.75% | 1.603 | 1.621 | 39261 / 21447 | 0.333 / 0.307 | FAIL |
| p512_c4 | BA | -70.55% | -63.26% | 1.595 | 1.643 | 39261 / 21447 | 0.333 / 0.307 | FAIL |
| p512_c8 | AB | -59.89% | -48.40% | 1.364 | 1.730 | 39725 / 21797 | 0.358 / 0.359 | FAIL |
| p512_c8 | BA | -62.23% | -50.63% | 1.386 | 1.763 | 39725 / 21795 | 0.358 / 0.359 | FAIL |
