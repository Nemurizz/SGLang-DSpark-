# One-cell TPOT recheck (2026-10-08)

This is validation/diagnosis tooling, not an inference optimization. The serving
checkout remains at `dspark-stable-fa4-fixed64-matrix128-passed-20261007` / `0577109`.
The tooling branch is based on `exp/paired-cache-long-validation` so existing
qualified policy, streaming, safety and reference helpers are available.

The AutoDL serving checkout contains pre-existing UNTRACKED framework dependencies.
Do not delete, add, overwrite or assume a new worktree includes those dependencies.
The templates refer to the preserved original checkout, models, SPS and h32 oracle;
this tooling worktree is NOT a standalone runnable model/framework distribution.

## Collector deployment

NVIDIA CUDA 13.2 Update 2 lists Nsight Systems 2025.6.3.541 with Linux driver
595.71.05, matching the current machine:
https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html

Official package:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb

Package SHA256:
`3aae3ce86316c07c9d743a0af009a3a8a1819a2f1f016487989e495173d0ce0d`

The package is extracted into an independent directory under `/root/autodl-fs`;
only the target CLI is invoked by absolute path. No dpkg installation, global PATH
or symlink change, driver/CUDA/runtime/library upgrade. Old nsys remains available.
NVIDIA documents standalone target CLI use:
https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#optional-setting-up-the-cli

`nsys_smoke.py` runs sixteen known FP32 matmuls inside sixteen NVTX ranges,
checks exact output, and refuses an occupied GPU. `audit_nsys_collector.py` checks
bounded kernel/NVTX counts, PID-qualified launch correlation and CUPTI 13.2.
It preserves all diagnostics: successful bounded coverage does NOT establish that
events outside the selected capture were collected, or that profiling is overhead-free.

## Unprofiled timing

`steady_tpot.py --template <h34-folder> --output <NEW-folder>` runs only p2048_c1,
128 output tokens, original qualified fixed64 numerical profile, ALL graphs OFF.
It checks sources, packages, driver/GPU, model metadata, policy, SPS, immutable
oracles, all output tokens, all five acceptance fields, cache geometry and retractions.
No diagnostic/NVTX/work hook or profiler is installed in the timing server.
The already-qualified numerical policy/attestation remains unchanged.

The protocol is persisted BEFORE observations:

- At least 24 and at most 64 warmup requests, groups of eight.
- Last three warmup group means spread at most 0.5%, each p95/p50 at most 1.02.
- Two continuous full/full windows of 64 requests each in the same server.
- Noise admission requires mean spread at most 0.5%, p95 spread at most 1%.
  These are symmetric BASELINE NOISE criteria, not candidate tolerances.
- Only after control admission: full/paired followed by paired/full, fresh owned
  server per arm, 64 steady requests per arm, same warmup admission.
- Candidate p95 tolerance remains ZERO for TTFT, TPOT and E2E. Stop immediately
  after a failed completed pair; no matrix expansion, retries or promotion.
- Keep every cold/warmup/steady record. Never delete slow observations, select
  a fast steady subset, or rerun a failed benchmark until it passes.

## Diagnostic trace

`diagnose_tpot.py` accepts an explicit `--nsys-bin` and `--nsys-trace cuda-sw|cuda`.
`cuda` on this machine selects HES/hardware trace; verify the actual diagnostics,
do not assume hardware tracing on other machines. Both profiles are bounded
full-only captures, 24 fixed settle requests, 32 diagnostic requests, no cache reuse.
`--flush-on-cudaprofilerstop=true` is explicit. No NCU is used or enabled.

`analyze_tpot_trace.py` attributes kernels using NVTX decode identity and launch
correlation, verifies 40 actual worker decode calls per request and geometry.
Response acceptance reports 39 verifies; never discard the extra execution or
pretend it is a paired-cache-specific difference. GPU kernel sums, GPU spans,
overlapping host spans and end-to-end latency are DIFFERENT quantities.
Profiled TTFT/TPOT must never be used to promote an optimization.

`decompose_baseline_tail.py` computes the kernel interval union and the span not
covered by this decode's kernels. It correlates those quantities with profiled
TPOT only. Fastest/slowest quartiles are descriptive diagnostics, NOT selected
benchmark populations. Nested host/API spans must not be added as exclusive
costs. Host values are normalized by 40 actual decode executions: prefill/cache
entries still execute once per request and are NOT per-decode method calls.

In h43 hardware trace, fastest/slowest quartiles have GEMM costs of
55.5709 / 55.5693 ms per actual decode, but kernel-uncovered spans of
4.0917 / 7.8671 ms. Kernel count/geometry is identical. This points subsequent
diagnosis toward dispatch/synchronization between kernels rather than claiming
GEMM arithmetic caused those profiled tails. It does NOT causally attribute the
uninstrumented h41 p95 spikes: profiler overhead is visible in both trace modes.

`summarize_nsys_recheck.py` reproduces this machine's h40-h43 artifact audit; its
fixed paths intentionally describe this saved run, not a generic benchmark CLI.

## Outcome contract

The h41 full/full control failed noise admission. No paired candidate timing
started. This does not revoke the original h34 p95 failure or prove cache gain.
Collector compatibility, correctness and performance stability are separate gates.
If a cause remains unproven, report it as unproven. Do not start GEMM, graph or
quantization experiments as part of this recheck. Stop owned services; keep the
qualified baseline tag, old references, main, existing untracked files and all
failed observations intact.
