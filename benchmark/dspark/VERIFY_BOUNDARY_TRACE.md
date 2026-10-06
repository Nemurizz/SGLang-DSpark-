# Eager verify boundary observation — 2026-10-07

Cache performance remains **UNPROVEN**. This is a correctness diagnostic,
not a performance experiment or an inference fix.

## Implementation and validation

Branch `exp/verify-boundary-trace`, based on the benchmark-only branch:

- `6acc654`: opt-in observer through the existing `--forward-hooks` API;
  only benchmark files changed, no inference monkeypatch or tensor replacement.
- `606f90f`: resolve actual ragged/fixed verify geometry, since target-verify
  does not populate ordinary extend lengths.
- 35 remote CPU tests passed. Local metadata tests passed with 2 torch tests
  skipped because local torch is unavailable.

The observer is off by default and requires an independent oracle and disabled
graphs. It identifies the prompt from a complete cache-off prefill, clears
identity on pool reuse, selects only a committed anchor, validates live and
padded geometry, and copies one final-hidden/raw-logit row to CPU. Snapshots
are bounded, use exclusive filenames, and never change the model output.
There is no teacher-forcing override; prefix equality is checked against saved
natural greedy responses. The observer can perturb scheduling and must not be
enabled in performance runs.

## Runs

- h10 (`dspark-batch-dspark-fi-boundary-20261007-h10`): observer stopped with
  a TypeError from an absent extend-length field; zero usable captures. It
  is not evidence of an inference-kernel fault. Service stopped and the
  baseline checkout was restored before repairing the benchmark observer.
- h11 (`dspark-batch-dspark-fi-boundary-20261007-h11`, revision `606f90f`):
  reproduced the original variant 3 concurrent divergence at output index 6.
  23 completed response rows and 3 bounded tensor captures were saved;
  pending requests were canceled and the owned server stopped on failure.

The focused request has the same prompt SHA, first 6 committed output tokens,
anchor token 47116, and anchor position 2053 in reference, serial repeat,
and concurrent observations. Correct/proposed/verify counts remain 26/77/11,
with histogram `[3,1,2,1,3,0,1]` in all three completed responses.

| Capture | Batch | Actual live/padded rows | Focus row | Raw logits 16953 / 1714 | Raw argmax |
|---|---:|---:|---:|---|---:|
| Reference | 1 | 8 / 8 | 0 | 21.625 / 21.5 | 16953 |
| Serial repeat | 1 | 8 / 8 | 0 | 21.625 / 21.5 | 16953 |
| Concurrent | 8 | 64 / 64 | 24 (request slot 3) | 21.5 / 21.5 | 1714 |

Reference versus serial-repeat final hidden and full-vocabulary raw logits
are bitwise equal. Reference versus concurrent:

- final hidden: 4357/5120 changed elements, max absolute delta 0.5,
  mean absolute delta 0.0102827800;
- raw logits: 105798/151936 changed elements, max absolute delta 0.125,
  mean absolute delta 0.0187130403.

The score-order change exists at the logits-processor output **before DSpark
compact scatter and acceptance**, and the LM-head input has already changed.
Thus a scatter/sampling-only fix is insufficient to eliminate the observed
upstream difference. This does not identify the responsible layer or prove
LM-head dispatch is invariant; there may be additional downstream differences.
Prefill-produced KV differences, earlier verification/commit history, attention,
GEMM and normalization remain possible upstream causes. CPU/device seq-length
differences in the snapshots are not independently treated as bugs: the verify
path deliberately prepares separate attention geometry.

## Reproduction and evidence

Use the h8 command in `BATCH_DIAGNOSTICS.md` from the observer branch, adding:

```bash
--verify-trace-output-index 6 --verify-trace-variant 3
```

Choose a new output directory; never overwrite old evidence. The results
include `verify_trace/*.json` (mapping/geometry/SHA) and `*.pt` (only one hidden
and one raw-logit row), independent-oracle manifest, raw responses, launch
configuration, logs, stop status and shutdown status. A separate CPU analysis
compares saved tensors with SHA checks; it never loads model weights or uses
GPU execution. The natural prefix and phase mapping are validated separately.

Next: locate the earliest upstream difference with a few selected layer/attention
boundaries and distinguish pre-existing KV divergence from the current verify
forward. Do not rerun full cache A/B until independent-oracle correctness passes.
If high paired-hit steady-state A/B then has no repeatable useful improvement,
stop cache work and choose a GEMM, graph or quantization experiment from measured
critical-path evidence. Each direction stays separate and revertible.

End state: `exp/reuse-compact-index@a61e0bb`; all original 8357 untracked files
preserved, no main changes or pushes, GPU idle, diagnostic service stopped.
