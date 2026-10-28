# Semantic strict A/B comparison (2026-10-29)

This experiment compares ordinary AR, DSpark full verify, and DSpark dynamic verify under the same request plan. Only the verification path changes. The result is an inference-path baseline, not an answer-quality benchmark or a production capacity promise.

## Protocol

- Model/runtime: Qwen3-14B BF16 on the DSpark/SGLang checkout; one server process per cell, restarted between arms.
- Prompts: 12 fixed prompts (4 code generation, 4 mathematics/reasoning, 4 open-domain QA). Every request uses the same Qwen3 tokenizer and is padded/truncated to exactly 512 input tokens; `max_new_tokens=128`.
- Arrival/concurrency: deterministic 2.0 requests/s (0.5 s start interval), `c=1/4/8`; 12 requests per arm/cell.
- Common switches: cache/radix/FP8/CUDA Graph disabled, BF16, deterministic seed, same port and same request order.
- AR: no speculative arguments. Full: `SGLANG_RAGGED_VERIFY_MODE=static`. Dynamic: `SGLANG_RAGGED_VERIFY_MODE=compact` with the SPS manifest.
- Reported latency: TTFT, TPOT, E2E; p95 is the empirical percentile over 12 responses. Server metrics are aggregated from decode-step records.

## Overall results

Units: ms; each cell is `mean / p95`. Acceptance is the DSpark verifier draft acceptance rate. Ordinary AR has no verifier field. Planner CPU is the host-side time of each dynamic budget decision; full verify has no dynamic planner and is N/A.

| path | c | acceptance | planner CPU | GPU step | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | — | — | — | 34328.1 / 61853.1 | 51.45 / 53.99 | 40862.0 / 68338.6 |
| ar | 4 | — | — | — | 4811.8 / 9460.8 | 51.00 / 51.58 | 11289.3 / 15928.1 |
| ar | 8 | — | — | — | 1076.7 / 2982.0 | 51.69 / 52.64 | 7641.2 / 9410.3 |
| full | 1 | 1.000 / 1.000 | — | 59.94 / 63.12 | 3330.7 / 5889.9 | 7.35 / 7.65 | 4264.6 / 6814.5 |
| full | 4 | 1.000 / 1.000 | — | 58.17 / 63.16 | 802.7 / 1617.8 | 8.93 / 8.32 | 1936.9 / 2991.3 |
| full | 8 | 1.000 / 1.000 | — | 59.13 / 63.22 | 580.2 / 1677.0 | 9.40 / 8.98 | 1774.6 / 3114.9 |
| dynamic | 1 | 1.000 / 1.000 | 0.176 / 0.201 | 63.97 / 65.28 | 3744.1 / 6669.7 | 8.06 / 8.19 | 4767.2 / 7647.1 |
| dynamic | 4 | 1.000 / 1.000 | 0.189 / 0.232 | 60.94 / 64.97 | 834.8 / 1612.6 | 9.53 / 8.95 | 2045.1 / 3080.0 |
| dynamic | 8 | 1.000 / 1.000 | 0.186 / 0.254 | 60.63 / 65.08 | 547.1 / 1627.1 | 9.68 / 9.20 | 1776.7 / 3094.0 |

CPU-side decode step and verifier detail (mean / p95):

| path | c | CPU step | target verify GPU | budget | record accept length |
|---|---:|---:|---:|---:|---:|
| ar | 1 | — | — | — | — |
| ar | 4 | — | — | — | — |
| ar | 8 | — | — | — | — |
| full | 1 | 58.36 / 63.17 | 49.34 / 51.99 | — | 8.0 / 8.0 |
| full | 4 | 55.03 / 63.11 | 47.66 / 51.92 | — | 8.0 / 8.0 |
| full | 8 | 55.47 / 63.16 | 48.44 / 52.07 | — | 8.0 / 8.0 |
| dynamic | 1 | 63.93 / 67.68 | 52.55 / 53.69 | 7.0 / 7.0 | 8.0 / 8.0 |
| dynamic | 4 | 60.73 / 68.30 | 49.89 / 53.49 | 21.0 / 28.0 | 8.0 / 8.0 |
| dynamic | 8 | 60.29 / 64.97 | 49.69 / 53.54 | 20.5 / 42.0 | 8.0 / 8.0 |

## Category view

The category table reports E2E and acceptance for code, math, and open QA. This run did not apply exact-match, unit-test, or LLM-judge scoring, so acceptance=1.0 must not be read as answer-quality improvement.

| c | category | AR E2E | full E2E | dynamic E2E | full acc | dynamic acc |
|---:|---|---:|---:|---:|---:|---:|
| 1 | code | 16430.5 / 19578.0 | 1993.1 / 2289.7 | 2160.4 / 2491.9 | 1.000 | 1.000 |
| 1 | math | 40891.0 / 43921.4 | 4269.1 / 4551.7 | 4804.5 / 5136.2 | 1.000 | 1.000 |
| 1 | open_qa | 65264.7 / 68338.6 | 6531.7 / 6814.5 | 7336.8 / 7647.1 | 1.000 | 1.000 |
| 4 | code | 6678.3 / 6672.8 | 2658.3 / 2991.3 | 2740.2 / 3080.0 | 1.000 | 1.000 |
| 4 | math | 11317.1 / 11307.6 | 1892.3 / 2237.6 | 2015.3 / 2354.5 | 1.000 | 1.000 |
| 4 | open_qa | 15872.4 / 15928.1 | 1260.1 / 1425.9 | 1379.8 / 1593.2 | 1.000 | 1.000 |
| 8 | code | 6815.2 / 6828.5 | 2796.9 / 3114.9 | 2800.6 / 3094.0 | 1.000 | 1.000 |
| 8 | math | 6724.9 / 6725.0 | 1329.4 / 1226.7 | 1320.7 / 1233.3 | 1.000 | 1.000 |
| 8 | open_qa | 9383.3 / 9410.3 | 1197.4 / 1233.4 | 1208.9 / 1251.4 | 1.000 | 1.000 |

## Findings

1. **DSpark gains are concentrated in decode throughput.** With this fixed-shape 2 req/s load, full/dynamic TPOT is about 7-10 ms while AR is about 51-52 ms. At c=4, full E2E is 1936.9/2991.3 ms and dynamic is 2045.1/3080.0 ms; AR is 11289.3/15928.1 ms.
2. **Dynamic planning has a fixed low-concurrency cost.** At c=1, dynamic planner CPU is 0.176 ms mean and 0.201 ms p95; compact GPU step averages 63.97 ms versus 59.94 ms for full. Dynamic E2E is 4767.2/7647.1 ms versus 4264.6/6814.5 ms for full.
3. **At higher concurrency the dynamic budget varies.** The budget mean/p95 is 21/28 at c=4 and 20.5/42 at c=8. The planner is reacting to load, but this run cannot prove semantic quality or real token savings because acceptance is saturated at 1.0.
4. **All three semantic categories have acceptance=1.0.** The fixed token shape, padding, and current draft/target pair made this a systems-path baseline, not a discriminative semantic benchmark.
5. **A production conclusion needs a quality loop.** Use real prompt length distributions, record position-wise survival probability and actual accepted tokens, run code unit tests, math answer checks, and human/LLM judging for open QA, and repeat at low/mid/saturated arrival rates.

## Reproduction files

- `run_semantic_suite.py`: fixed-prompt runner and server lifecycle.
- `overall_summary.json`: 9 cells x 12 responses with server-side step aggregates.
- `category_summary.json`: 27 category/cell/path aggregates.
- `ar_c*/`, `full_c*/`, `dynamic_c*/`: raw response records and server logs.
- Planner instrumentation records `planner_cpu_ms` in `VerifyBudgetDecision`, `DecodeStepObservation`, and the info dumper; see `INSTRUMENTATION.md`.

## Limitations

- n=12 per path/concurrency cell is a smoke/baseline sample, not a confidence interval.
- Input/output lengths are fixed (512/128); cache, FP8 and CUDA Graph are disabled, so this isolates verifier/planner behavior rather than the best production configuration.
- Prompts are semantic categories but were padded to a fixed token shape and no answer-quality grader was run. Acceptance is a systems signal only.
- Full verify planner CPU is N/A by design; AR has no verifier/planner/server-step fields.
