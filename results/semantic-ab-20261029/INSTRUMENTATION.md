# Planner CPU instrumentation

The semantic run records the host-side time spent by the dynamic budget planner. The timing starts immediately before `HostConfidenceBudgetPlanner.compute_budget` evaluates the request state and ends when the `VerifyBudgetDecision` is returned. The value is carried through `DecodeStepObservation`, `_PendingStep`, and `DecodeStepRecord`, then emitted by `DsparkInfoDumper` into the existing `dspark_info_record` stream.

The patch is included as `planner_cpu_instrumentation.patch`. The planner hunk also contains the nearby per-request max-length setter that was already present in the checked source; the new timing fields themselves change only observability data flow and do not change budget thresholds, verifier decisions, or model weights. Full verify is intentionally reported as N/A because it bypasses the dynamic planner. A successful `py_compile` check was run on both modified Python files before the A/B run.
