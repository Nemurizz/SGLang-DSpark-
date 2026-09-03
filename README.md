# SGLang / DSpark Dynamic Verification Scheduling

A reproducible engineering snapshot for profiling and optimizing dynamic verification in SGLang's DSpark speculative decoder.

## What changed

- Runtime A/B control switches automatic and fixed verification without restarting the server.
- Per-step observability records candidate-token counts, the executed verification path, CUDA Graph state, and split CPU/GPU timings.
- Profiling localized roughly 0.25 ms/step of added latency to CPU-side budget computation.
- A layout-scheduling hypothesis was rejected quickly: the cache hit rate exceeded 99%, but the saving was only about 19 us/step.
- Reusing immutable performance-cost data reduced the planner CPU micro-path from 90.322 us to 39.205 us (56.6%) while preserving budget decisions.
- The final dynamic-vs-fixed end-to-end overhead converged to about 1%.

## Validation

The recorded engineering run completed 91 tests and an ABBA experiment with 4 x 256 requests. Fixed-token output checks passed in every arm. Raw machine-readable evidence is under `results/`.

- `sglang/` contains the changed source and tests at their original paths.
- `sglang-dspark-dynamic-verification.patch` applies the complete change set to the matching SGLang checkout.
- `results/sps-probe/` contains additive steps-per-second probe data.
- `results/verify-path/` contains path-selection and CUDA Graph observations.
- `results/abba-4x256/` contains the controlled end-to-end A/B/B/A run.

## Key implementation points

`dspark_observability.py` provides structured low-overhead records. `dspark_worker_v2.py` emits the chosen path and timing information. `dspark_planner.py` implements budget selection, uniform-layout caching, and cached additive step costs. The request structures and environment controls in `io_struct.py` and `environ.py` expose runtime switching.

## Applying the patch

```bash
cd /path/to/compatible/sglang
git apply --check /path/to/sglang-dspark-dynamic-verification.patch
git apply /path/to/sglang-dspark-dynamic-verification.patch
```

The snapshot was syntax-checked with `python -m py_compile`. The remote environment no longer had pytest installed when this repository was assembled, so it preserves the test files and recorded 91-test result rather than claiming a fresh rerun.

## Hardware context

Measurements used an NVIDIA RTX PRO 6000 Blackwell Server Edition. Absolute timings depend on SGLang revision, CUDA/PyTorch stack, model, batch mix, and CUDA Graph configuration.

## Upstream

Based on [SGLang](https://github.com/sgl-project/sglang). See `LICENSE` for the upstream license.
