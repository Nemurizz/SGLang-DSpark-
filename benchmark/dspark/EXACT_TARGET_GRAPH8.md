# Target-only exact 1x8 graph experiment

Independent branch, not stacked with either rejected residual norm candidate.
Existing target DecodeCudaGraphRunner full backend, original exact geometry
guard unchanged and additionally restricted to one request, eight real tokens,
host verify-lens [8]. Catalog [8] only: do not introduce fine tiers that change
the original full-width SPS verification path. Mismatch/unknown geometry eager.
Buffer capacity remains bs1/4/8 to handle concurrent eager fallbacks safely.
Prefill graph OFF; DSpark draft graph OFF, folded proposal OFF. Existing target
graph path includes its existing epilogue; no epilogue kernel written/changed.
FA4 original fixed64x64 host policy, split1, BF16 KV/page1, original registered
batch-invariant GEMM/reductions retained. New runtime attestation explicitly
distinguishes graph experiment from qualified graph-OFF baseline; never weaken
the old validator or mutate the serving checkout. Frozen h32 tokens and five
acceptance fields still required, including serial/c4/c8 before timing.

Theory: norm-only catalog kept 180 input copies per decode and 800 private
entries (+5706 MiB), delivering only 0.0634% mean E2E with worse tails. One
whole-target graph removes per-norm staging and captures more dispatches;
plausible >1% critical-path space, NOT a proved gain. All prescribed fresh
control, warmup, unprofiled AB/BA and zero-p95-regression gates remain. Scope
p2048 c1 output128 only. Stop on any correctness/acceptance/illegal access.

Primary background:
https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell/
https://github.com/sgl-project/sglang/pull/37300
Neither source justifies relaxing geometry or predicts our E2E benefit.
