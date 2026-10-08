# Fixed verify-all: bounded policy experiment (2026-10-08)

Candidate is NOT equivalent to adaptive verify when the CPU budget is full:
the GPU scheduler can still filter current survival probabilities below EPS.
This experiment explicitly omits the SPS table, reuses the existing uniform
compact layout, and disables only the unused host budget planner at startup.
Confidence head, confidence relay, sampler, KV write/readiness, acceptance and
commit code remain intact. No live positions/KV locations/prefixes are cached.
No kernel, graph, quantization, or paired-cache changes are combined here.

The opt-in `verify_all_policy_hook` is installed before planner initialization
using the existing forward-hook factory. It changes process-local configuration;
it is not a read-only observer. No existing tracked/untracked inference file is
overwritten. Service exit rolls it back. Factory fails closed unless compact,
TP1/PP1/DP1 without DP attention, gamma7, width8, page1, deterministic inference,
graphs OFF, no dynamic length/capture/collection and SPS path absent.

Frozen arithmetic is the separately qualified BF16 FA4 fixed64 numerical policy.
All 128 output token IDs and five acceptance fields MUST equal immutable h32
references. No new oracle is generated. GPU UUID migration is recorded explicitly
and needs a new full-control correctness check; same GPU name, driver, packages,
sources and model JSON hashes are mandatory. This does not requalify other cells.

Protocol is frozen in `verify_all_ab.PROTOCOL` BEFORE observing results:

- Timing: p2048 c1 output128, no profiler or per-step diagnostic hooks.
- Correctness: p2048 c1/c4/c8, every variant serial then two concurrent waves,
  identically for EVERY arm. All warmup/timing rows are also checked.
- Warmup: 24..64 requests in groups8, last3 TPOT means spread <=0.5%, group
  p95/p50 <=1.02. Preserve all records; failure ends the attempt.
- Control: two consecutive windows64, TPOT and E2E mean spread <=0.5%, p95
  spread <=1%. Candidate never starts on failed control admission.
- Four fresh-server arms: adaptive/fixed/fixed/adaptive, 64 steady requests each.
- Candidate p95 TTFT/TPOT/E2E tolerance ZERO in each order. Immediate stop.
- Mean E2E improvement must be >=1% in BOTH orders; end-to-end control drift
  must also pass the same noise limits. Limited cell evidence, not matrix proof.
- Two-hour wall budget includes server startup, correctness, warmup and shutdown.
  No request retries, no enlarged matrices, no failed-window discard/rerun.

Run from the independent tooling checkout, serving the ORIGINAL dependency tree:

```bash
cd /root/autodl-fs/dspark-verify-all-tools-20261008-h44/benchmark/dspark
/root/conda-envs/sglang/bin/python -m unittest test_verify_all_policy test_verify_all_ab
/root/conda-envs/sglang/bin/python verify_all_ab.py \
  --template /root/autodl-tmp/results/dspark-cache-balanced-ab-20261007-h34 \
  --output /root/autodl-fs/dspark-verify-all-ab-20261008-h45 \
  --budget-origin /root/autodl-fs/dspark-verify-all-ab-20261008-h44/manifest.json
```

Implementation inspiration (not transferable performance evidence):
[Modal host-overhead article](https://modal.com/blog/host-overhead-inference-efficiency)
and [SGLang Spec V2 article](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).

Status before execution: benefit UNPROVEN. Service checkout remains at
`dspark-stable-fa4-fixed64-matrix128-passed-20261007` (`0577109`).

Pre-candidate protocol repair: h44 was stopped during control acquisition,
before any candidate ran and before a control admission decision. Its manifest,
logs and every request are retained. Review caught asymmetric conditioning
(initial version ran correctness only on control0 and first candidate). h45
uses identical correctness conditioning on every arm. Same thresholds and
original two-hour deadline. This is not a failed-control admission retry.
