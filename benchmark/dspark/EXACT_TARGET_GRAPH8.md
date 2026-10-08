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

## h54 diagnostic audit / integration repair

h54 preserved 68 requests and passed serial/c4/c8 frozen tokens and acceptance,
but ZERO graph admissions and ZERO fallbacks: it never created a target graph.
The inherited launch environment had SGLANG_DSPARK_RAGGED_VERIFY_GRAPH=0.
The startup warning combined disabled opt-in and backend capability; source
inspection confirms FlashAttentionBackend declares supports_ragged_verify_graph=True.
Also regular forward hooks are registered AFTER graph capture, and the default
slot policy with max_bs=8 captures eight rows, not one row with eight tokens.
This is NO-EFFECT evidence, not graph correctness or performance evidence.

Repair the isolated launch harness: explicit ragged opt-in, process-local policy
installation immediately before original target capture, catalog [8], capture
slots exactly one while preserving buffer max_bs=8. Spawned workers use a dedicated
launcher. Backend capability and the ORIGINAL admission guard are never overridden.
Assert actual capture layout and draft OFF before requests; require positive
admissions only at bs1 before completing diagnostics or entering timing.
No unprofiled A/B has failed or been retried for this candidate.
