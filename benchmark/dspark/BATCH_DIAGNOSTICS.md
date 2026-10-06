# Batch invariance investigation — 2026-10-07

Cache A/B remains incomplete and performance benefit remains **UNPROVEN**.
Do not run paired/target-only measurements past a failing full-prefill gate.
All experiments below use the exact saved `p2048_c8` dataset, 32 output tokens,
greedy PyTorch sampling, deterministic inference, no prefix cache, and all CUDA
graphs disabled. Draft attention is FlashInfer and target decode is FA4.

## Evidence

| Run | Only changed factor | Result |
|---|---|---|
| h5 | Remove DSpark; ordinary target FlashInfer prefill | 8 references plus 3 × (8 serial + 8 concurrent) requests passed. All 8 ordinary-target references also match the earlier h4 DSpark serial references. |
| h6 | DSpark target prefill/verify uses FA4 | Stopped on concurrent variant 2, output index 6: serial 1714, concurrent 16953. Backend change is not a sufficient fix. |
| h7 | Candidate eager FlashInfer verify `fixed_split_size` propagation | 3 serial/concurrent rounds internally stable, but post-run independent h5 oracle check found 3/8 variants different, in 21/56 saved requests. Candidate rejected and rolled back. |
| h8 | Original DSpark + return top-3 logprobs; independent h5 oracle | Reproduced h4 concurrent variant 3 divergence at output index 6, 16953 → 1714. |
| h9 | Ordinary target with built-in Triton attention, independent h5 oracle | Stopped already at serial reference variant 3, output index 30: 421 → 979. Not adopted; no new Triton GEMM was written. |

h7 status records its original **within-profile-only** pass. It is not an
oracle pass or an approved configuration. The later oracle comparison rejects
it. The diagnostic runner now accepts `--oracle-jsonl` and checks even new
serial references, so a candidate cannot silently replace its expected output.
The oracle's adjacent dataset must match the selected prompts exactly.

For h8 variant 3, the top logprobs at index 6 are:

| Phase | token 16953 | token 1714 | chosen token |
|---|---:|---:|---:|
| serial reference / serial repeat | -1.4856604338 | -1.6106604338 | 16953 |
| concurrent | -1.5488902330 | -1.5488902330 | 1714 |

The serial gap is 0.125; concurrency changes it to an exact tie. Correct drafts,
proposed drafts, and verify count remain 26 / 77 / 11 in all three responses.
The eager production accept paths use `torch.argmax`; a tie selects the lower
token ID. Therefore this failure has already changed target score ordering
before acceptance. It is not explained merely by a varying acceptance count.
These observations do not yet identify the responsible attention, GEMM,
normalization, KV, or metadata operation. Do not change tie-breaking to hide it.

## Isolated branches and rollback

- `exp/paired-cache-ab-harness`: benchmark/diagnostic changes only, no inference changes.
  - `652505e`: isolate ordinary target versus DSpark.
  - `4f8a8d6`: independent ordinary-target oracle, top logprobs, service-environment PATH.
  - `d2e4471`: built-in Triton attention control, not an optimization claim.
- `exp/verify-deterministic-split@887e9e1`: rejected eager-only candidate, retained
  for inspection; **not merged**. Unit tests prove parameter forwarding, not
  output preservation. Its CUDA graph path was not modified or validated.
- The working checkout is restored to `exp/reuse-compact-index@a61e0bb`, tag
  `dspark-optimization-history`, after experiments. No main changes or pushes.
- All 8357 pre-existing untracked baseline files are preserved. Services stop
  on each failure; no experiment is left serving.

## Reproduce the original mismatch with score evidence

Use the service Python. This diagnostic stops at the first discrepancy and is
not a performance experiment:

```bash
/root/conda-envs/sglang/bin/python benchmark/dspark/dspark_batch_diagnostic.py \
  --repo /root/autodl-tmp/src/sglang-main \
  --model /root/autodl-tmp/models/Qwen3-14B \
  --draft /root/autodl-tmp/models/dspark_qwen3_14b_block7 \
  --sps-table /root/autodl-tmp/results/sps-probes/dspark14-sps-probe-20260830-210115/qwen3-14b-rtxpro6000-additive-sps.json \
  --dataset /root/autodl-tmp/results/dspark-cache-ab-deterministic-control-20261007-h4/dataset.json \
  --oracle-jsonl /root/autodl-tmp/results/dspark-batch-target-fi-20261007-h5/requests.jsonl \
  --profile dspark-fi --logprob-topk 3 --repeats 1 \
  --output /root/autodl-tmp/results/dspark-batch-NEW-UNIQUE-ID
```

The benchmark files live on the harness branch, not on the restored baseline.
Each result folder retains manifest/revision, exact dataset, launch/runtime
configuration, raw streaming responses, service logs and shutdown status.

## Dependencies and next checks

`pytest` and `pytest-asyncio` were installed in the service environment through
the configured Aliyun mirror. `ninja` was already installed there, but an
absolute Python path alone does not put its executable on PATH. The harness
now prepends that environment's bin directory. Earlier stacked-KV tests failed
for missing `ninja` on PATH, not a numerical parity failure; they passed after
correcting PATH. GPU kernel parity and stacked-KV parity tests do not prove
whole-model numerical invariance.

Next: an opt-in, non-mutating activation/logit trace comparing the same request
under serial and concurrent target verify, retaining request-to-row mapping,
attention geometry and the earliest layer difference. Validate fixes against
both the independent target oracle and original DSpark references before
resuming the 12-cell cache matrix. Keep all geometry guards, graph-off default,
and acceptance/output gates intact. Cache counters, recovery plans and actual
worker fallback counts are distinct; absent metrics remain unavailable.
