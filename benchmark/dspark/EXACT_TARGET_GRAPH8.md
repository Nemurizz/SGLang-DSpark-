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

## h55/h56: capture alone still not runtime replay

h55 correctly captured 1x8 (0.126953 GiB) and kept draft graph OFF. 68 saved
requests still matched the frozen oracle, but the two profiled p2048 requests
(80 actual decode steps) contain ZERO cudaGraphLaunch APIs. The guard reported
only two admissions outside that trace and 3616 fallbacks. h56 host-only guard
facts show the exact rejection: runtime bs1, tier8, hidden FULL, but host length
and live total both None, while capture has [8]. Do not classify these as successful
graph correctness or E2E gains. No unprofiled A/B was attempted. The initial
startup-only admission test was insufficient; actual execute counters and a
positive steady-window delta are now required.

## Separate host-mirror bridge candidate

Branch exp/exact-target-graph8-host-mirror adds one guarded CURRENT D2H read for
device-only bs1/tier8 layouts. msgspec immutable replacement preserves every
device tensor/pointer and adds only the actual ordered CPU length and its sum;
the forward batch's spec layout is updated to the replacement. The original
guard is called unchanged and still rejects partial lengths, missing captures,
multirow geometry and hidden-mode mismatches. No bucket-derived guess or stale
mirror; no change to scheduling, GEMM, reduction or target attention arithmetic.
The D2H synchronizes; that cost is included in unprofiled timing. The old guard
already performed a device sum().item() for these unknown totals, so a bridge
could exchange that wait for a host length AND enable replay, but gain remains
UNPROVEN. Do not claim a free read or assume synchronization removal.

## h57 actual replay and h58 unprofiled result: tested-cell PASS; STOP

Measurement commit: 74666bf2ca8a8dd18ce4b3a93433b992be94c8cc1.
Candidate remains opt-in on exp/exact-target-graph8-host-mirror. Original serving
checkout is detached 0577109, tracked clean; main remains 0c1f1eb. No new GitHub
push was performed. No further kernel, graph, GEMM or quantization experiment
was opened after this result.

h57: 68 checked requests including serial/c4/c8, all frozen tokens and five
acceptance fields matched. Two profiled p2048 requests: 40 real decode steps each,
80 cudaGraphLaunch APIs. Capture/runtime geometry remains bs1 / ordered [8].
Graph capture sampled allocation increment ~0.127 GiB. Profiling is mechanism
evidence only; the 61.969687 ms GPU span is NOT the performance claim.

h58: full prefill / radix and paired caches OFF in both arms, Qwen3-14B BF16,
SM120, original fixed64 FA4 and batch-invariant operators; prompt2048, output128,
concurrency1. Fresh two-window baseline noise admission; A-B-B-A with 64 steady
requests in every comparison arm, all 39 serial/c4/c8 correctness checks per
arm and bounded fixed warmup. No retry or steady-request discard.

| Order | Metric | Baseline mean (ms) | Candidate mean (ms) | Mean improvement | p95 improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| AB | TTFT | 256.7236 | 226.0683 | 11.94099% | 4.20755% |
| AB | TPOT | 18.9890 | 18.8120 | 0.93231% | 0.90850% |
| AB | E2E | 2668.5155 | 2615.3556 | 1.99212% | 1.30856% |
| BA | TTFT | 256.1672 | 233.1751 | 8.97544% | 3.39569% |
| BA | TPOT | 18.9847 | 18.8070 | 0.93618% | 0.95867% |
| BA | E2E | 2667.4172 | 2621.8601 | 1.70791% | 1.20600% |

Both orders pass >=1% mean E2E AND zero-regression TTFT/TPOT/E2E p95 guards.
Fresh baseline mean spread: TPOT0.02310%, E2E0.03736%. Counterbalanced A1/A4
mean drift: TPOT0.02274%, E2E0.04118%; p95 drift admission also passed.
The decision is specific to this workload and current instance, not a general
deployment qualification or a throughput/concurrency/paired-cache claim.

708 saved requests independently rechecked against frozen h32 oracle, including
195 prescribed correctness requests. Steady acceptance identical in every arm:
correct5760 / proposed17472, verify2496, rate0.32967032967032966. Each candidate
window's execute-counter delta is 2600, including one excluded presteady request;
actual execution only bs1. CPU policy tests12 + A/B protocol tests5 passed;
original ragged10/backend-capability2 passed before integration.

Device-wide steady memory samples (not allocator peak): baseline57005 MiB,
B2 57133 MiB, B3 56985 MiB. Do not claim a constant graph memory delta or a
memory optimization from these samples. Clock-event reasons were zero in all
steady samples. The narrower target graph avoids the previous norm catalog's
multi-GiB pool growth, but causal TTFT/memory subcomponent attribution is not
proved by this A/B. The claim belongs to the combined exact graph + real-length
bridge candidate, not the graph flag alone.

Frozen sources/oracle/tool digests passed at completion and in post-run recheck.
8357 existing untracked paths remain; no deletion was performed. Current NUL
pathname-list digest is 9077a67cf4e576a78b7750734c1e956defe85114739b9bb6eed68bc828708b7e.
An older human summary erroneously transcribed 65 characters (extra 7); the
post-run checker corrected that transcription, not any inference source or
measurement. This is not full-content qualification of every untracked file.
Owned services/listener drained, GPU process list empty. Default safe graph-OFF
baseline remains untouched. Per user request, STOP optimization here.
