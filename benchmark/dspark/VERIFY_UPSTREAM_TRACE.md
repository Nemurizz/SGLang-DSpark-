# Read-only upstream boundary isolation

This extends the default-off eager observer, not an inference fix or an
optimization. No backend, sampling, acceptance, graph geometry or arithmetic
policy is changed. Do not use observer timings as performance measurements.

`dspark_batch_diagnostic.py --verify-trace-output-index 6
--verify-trace-variant 3 --verify-trace-upstream
--verify-trace-max-captures 15` registers one shared hook on:

- `model.layers.0.self_attn.attn`: Q/K/V call arguments and returned attention
  output at the committed anchor; past layer-0 KV and current written KV.
- `model.layers.0`, `.19`, `.39`: returned hidden and residual, not aliased
  post-forward input tensors falsely described as pristine input.
- `logits_processor`: final hidden and raw logits as before.

Requires an independent oracle, disabled CUDA graphs, identical saved prompts,
cache-off complete prefill identity and runtime-authoritative live query lengths.
Past KV is ordered by the request's physical row map for positions `[0, anchor)`;
it excludes the current token and speculative future. CPU validation checks
mapping length, uniqueness, bounds and disjointness from current verify writes
before GPU gather. Only unquantized three-dimensional NHD rows are supported;
prefix length is bounded at 4096, and only layer 0 is captured.

Deduplication is per request epoch and stage, with at most 15 snapshots (three
five-stage forwards) in this reproduction. Observers return None, never mutate
model output/KV and never redefine the independent serial oracle. A token or
acceptance mismatch must stop the owned service and restore the known checkpoint.

Interpreting evidence: identical layer-0 Q/K/V and past KV with different
attention output narrows the cause to attention execution/planning. Different
past KV requires separating prompt KV from earlier generated-token KV before
assigning causality. Differences at a later boundary alone do not prove the
responsible operation or a usable fix.
