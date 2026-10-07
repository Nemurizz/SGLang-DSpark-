# Qualified long-output cache validation protocol

This extends the independently authorized fixed64 FA4 numerical baseline. It
does not introduce a GEMM, kernel, graph or quantization optimization. All graphs
remain off; original references and pre-existing untracked sources are retained.

## Prerequisite

`stable_baseline_gate.py --fixed-fa4-tile --cells all --output-tokens 128
--with-paired` must pass ordinary target, DSpark full prefill and real paired reuse
for all 12 cells, with exact tokens and all five DSpark acceptance fields. The
128-token references are frozen separately from the earlier 32-token references.

## Natural partial-recovery pressure

Use the unchanged 25% sidecar budget and an 8K shared-prefix/eight-suffix group.
Six 8K pressure prefixes with distinct first token exceed sidecar metadata
capacity, while the target KV capacity remains sufficient. Keep the shared root
recent via suffix 7, allowing older suffix metadata to be evicted naturally.
Never inject faults, fake readiness or bypass target-only guards. Full-prefill
controls are separately collected for the synthetic pressure prompts.

The default-off process-local observer wraps existing prefill/injection calls
without changing tensors or scheduler decisions. It records actual protected
prefix length, target prefill count and successful draft injection enqueue count.
Join worker records to completed responses by request ID; completion plus exact
tokens/acceptance validates asynchronous GPU execution. Require actual partial
recovery in serial, four and eight concurrent requests. Metadata plan counters
are retained separately; they are not execution or fallback counters.

This pressure gate covers 8K prefixes, not every possible recovery geometry.
The observer is disabled during timing; its instrumentation cost is not hidden
inside the cache performance result.

## Stable-state A/B screen

Same qualified numerical profile and frozen 128-token references in both arms.
Two paired blocks: A/B then B/A, with fresh owned services for each arm. Same
seeded cell order within each paired block, 64 measured requests per cell/arm/
block, waves at concurrency 1/4/8. Shape-specific priming and two settle waves are
excluded and saved separately. Use cumulative-token SSE, fresh HTTP connections,
no generation retries, device-wide memory/utilization/temperature at 100ms.

This is a **100% hot repeated-prefix workload**, not a mixed/cold production
benchmark, and does not contain an unsafe target-only cache arm. Require the hot
cached length to equal prompt length minus one. TTFT is first token-bearing SSE;
TPOT is amortized (last-first)/(N-1), not fabricated inter-token latencies.

Stop on any token/acceptance mismatch, illegal memory error or descriptive TTFT,
TPOT or end-to-end p95 regression. Regression tolerance is zero, as requested;
no threshold may be relaxed after observing the result. Stop as soon as a cell
has both arms, not after finishing the whole second arm. Preserve failed data
and roll back to the last qualified tag; a stopped run is not a completed A/B.

Report raw request counts, latency distributions, throughput, acceptance, cache
match counters/gauges and sampled device memory. Shadow target hit statistics
are pre-truncation match observations, not unique-request hit probabilities.
Timing recovery/fallback counters remain labeled **plans**; actual execution
coverage comes from the separate pressure gate, never assumed from missing data.

Prespecify a material hot-TTFT threshold of 5% and paired-wave cluster bootstrap
confidence intervals. With only two independent paired blocks, intervals are
conditional on the tested runs, not broad production guarantees. Additional
independent repetitions and realistic hit-rate workloads are needed for general
deployment claims. Baseline arithmetic/profile cost is not paired-cache gain.
