# Eager serial-reference partition correctness candidate

Default OFF: `SGLANG_DSPARK_VERIFY_SERIAL_REFERENCE=1`, or the harness flag
`--reference-verify-plans`. This is a correctness candidate, not proven faster.
It changes ONLY eager FlashInfer DSpark target-verify planning/dispatch. Prefill,
draft, sampling, acceptance, KV writes, compact/scatter and oracle remain unchanged.
No GEMM, CUDA graph or quantization experiments are involved.

Prerequisites are deliberately narrow: FlashInfer **0.6.17**, FA2, all CUDA graphs
disabled, BF16 Qwen3 (40 Q heads, 8 KV heads, head dim 128), NHD page size 1, TP1,
no sliding window, multimodal input or custom mask. Unsupported geometry fails
closed rather than silently using a different plan. No changes to exact graph
geometry guards or deterministic prefix-cache restrictions.

Evidence: the h13 layer-0 capture at the first differing committed token has
bitwise-equal Q/K/V, prior KV and current KV writes for B1/B8. The installed
automatic FA2 plan uses 128-token chunks for B1 and 412 for B8, producing 1955
different output elements. A focused replay exactly reproduces BOTH saved
automatic outputs; fixed 128 reproduces the original B1 output, whereas fixed
4096 produces a different B1 output. Neither is end-to-end performance evidence.

Preserve each request's original **B1** plan, not a workload-wide guessed constant.
For the checked head geometry and live query length 1..8, B1 has one Q tile.
FA2 CTA_TILE_Q is 16 for live Q length <=3 and 64 otherwise. The original B1
binary-search KV chunk rule therefore reduces to
`max(128, ceil(kv_length / floor(2*SM_count/8)))`, in pages (here one token/page).
Adjacent requests with equal `(chunk, CTA_TILE_Q)` share a wrapper. Different
geometries execute sequential groups; returned rows retain original order.

The candidate reads the original eager wrapper's actual Q/KV indptr, requires
live sum == runtime input row count and rejects padded/stale layouts. B1 retains
the original automatic wrapper, exactly. B>1 groups use fixed splits derived
from each B1 geometry and validate actual plan CTA/Q-row count/graph-off fields.
Existing KV is written once, before grouped attention. Group wrappers reuse
float workspace in stream order and have separate bounded per-group int buffers.

Tradeoffs: extra planner D2H, grouped dispatch, merge work and int workspace may
increase TPOT/p95 and memory. Keeping serial arithmetic is NOT itself a speedup.
Run independent target and original DSpark references, serial/C4/C8 token and
acceptance checks, then steady cache A/B. Any mismatch or adequate-sample p95
regression stops the owned service and restores a61e0bb; no promotion/push until
those gates pass. The known deterministic FlashInfer configuration still disables
prefix reuse, so a deterministic-only pass cannot demonstrate paired-cache gain.
