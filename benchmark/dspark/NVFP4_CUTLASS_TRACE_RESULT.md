# h65 bounded CUTLASS target trace

Evidence: `dspark-quant-cudnn-quality-20261009-h65/cutlass_trace`.
Nsight Systems 2025.6.3 software CUDA13.2 collector, graphs/cache off. Four
requests with 64 fixed output tokens each matched their own newly generated
quantized AR tokens. Prime/warmup/capture/closing requests are all preserved.

Observed 95,113 kernels, all correlated to runtime launches. Hook counters
recorded 10,400 native FP4 GEMMs and 10,400 activation quantization calls,
including the captured prefill/decode forwards. SQLite/statistics retained.
Collector emitted not-all-NVTX/CUDA-events caveats around bounded stop/cleanup;
do not remove these diagnostics or claim global completeness.

Within the recorded kernel span (6269.43 ms), GPU kernel busy union was 1015.16
ms (16.19%). Summed kernels: FP4 CUTLASS 616.79 ms, FA4 127.93 ms, BF16 dense head
101.84 ms, activation FP4 quantization 20.95 ms, other kernels 149.18 ms.
FP4 is 60.7% of summed kernel time, NOT 60.7% of end-to-end time.

M1 CPU-side native FP4 GEMM ranges: medians 195.50 us (packed K2560) and
170.20 us (packed K8704). Quantization medians 65.56 us (K5120) and 56.79 us
(K17408). Their summed wall time is substantial; CPU ranges include any waits
and profiler cost. Do not infer full unprofiled CPU utilization from NVTX.

No cudaMalloc/cudaFree API calls were observed in this warmed window. Installed
FlashInfer still has the small-workspace fallback source described in issue
4549, but PyTorch caching-allocator churn would not necessarily issue device
malloc/free. Hence the issue is NOT established as the local slowdown cause.

Decision: do not write a replacement GEMM or claim a kernel-only fix. Test
native backend/precision alternatives independently and retain quality plus
unprofiled equal-capacity stable ABBA as the success requirement.
