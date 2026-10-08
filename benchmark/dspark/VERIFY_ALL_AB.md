# Fixed verify-all: bounded policy experiment (2026-10-08)

Candidate is NOT equivalent to adaptive verify when the CPU budget is full:
the GPU scheduler can still filter current survival probabilities below EPS.
This experiment explicitly omits the SPS table, reuses the existing uniform
compact layout, and disables only the unused host budget planner at startup.
Confidence head, confidence relay, sampler, KV write/readiness, acceptance and
commit code remain intact. No live positions/KV locations/prefixes are cached.
No kernel, graph, quantization, or paired-cache changes are combined here.

The opt-in `verify_all_policy_hook` binds the already-constructed planner in the
startup forward-hook factory, BEFORE request processing/callback registration.
This version defers graph/hook setup until after worker construction. The exact
type/one-instance lookup is startup-only; no objects/tensors are scanned per step.
It changes process-local configuration;
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

Startup repair: h45 completed control0 windows and A1, then stopped because the
constructor interception was installed AFTER the planner already existed. No
candidate correctness/timing rows were produced. Runtime attestation caught this;
it is not a token/acceptance/p95 failure. The repair binds the unique existing
planner only while its uniform layout cache is empty. `--resume-startup-repair`
accepts ONLY this exact unmeasured failure, rechecks all immutable controls and
sources, keeps the original deadline, and resumes B1/B2/A2 without repeating
control acquisition. It cannot resume a measured correctness/performance failure.

```bash
/root/conda-envs/sglang/bin/python verify_all_ab.py \
  --template /root/autodl-tmp/results/dspark-cache-balanced-ab-20261007-h34 \
  --output /root/autodl-fs/dspark-verify-all-ab-20261008-h46 \
  --budget-origin /root/autodl-fs/dspark-verify-all-ab-20261008-h44/manifest.json \
  --resume-startup-repair /root/autodl-fs/dspark-verify-all-ab-20261008-h45
```

Source of lifecycle confirmation:
[upstream hook setup](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py).

h46 also stopped before requests: the factory executes inside graph setup before
`ModelRunner.decode_cuda_graph_runner` is assigned. The startup guard now allows
that unassigned field ONLY together with explicit args graph-off and independently
validated resolved PREFILL/DECODE DISABLED backends. An existing graph runner or
already-created layout remains rejected. No geometry guard is weakened. h47 is
the last startup attempt; another startup failure ends this candidate. h45
controls remain the sole immutable controls; h46 failure logs are retained.
