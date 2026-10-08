# h65 cuDNN route stopped, no quality or stable benefit claim

The installed native cuDNN FP4 backend passed 16 dimension/M duplicated-row
smoke checks, but that did not establish serving performance or model quality.
New real-task ordinary AR completed GSM8K219 (187 tokens), GSM8K42 (78) and
MATH752 (1749). The second long MATH item exceeded the unchanged 240-second
deadline while the scheduler continuously generated approximately 10 tokens/s.

This route, like h64 CUTLASS, was stopped automatically. Quantized DSpark,
384-item scored quality and stable A/B did not start. No illegal memory access
or completed-token disagreement was observed; SIGTERM/SIGQUIT in final logs
follow intentional owned-service cleanup, not a spontaneous scheduler failure.
The original BF16 baseline and source identity remained unchanged; GPU empty.

Do not raise deadlines, shorten the fixed output cap, exclude the long item or
reinterpret unconditioned AR timings as a DSpark A/B result. cuDNN is not a
proven performance fix. Preserve full evidence and the independent branch.

Next experiment is the conservative installed native W8A8 FP8 path from the
original BF16 parent, with no custom GEMM or dependency upgrade. Same quality
data/grader and safety/benefit gates. Do not reuse NVFP4 backend token oracles.
