"""Exact-shape, per-module/stream graph catalog of ORIGINAL residual RMSNorm.

Not a fused kernel: capture the unchanged callable, including its registered
batch-invariant mean and BF16 intermediate rounding. No attention/plans/KV.
Separate private pools; held inputs/outputs; copies and replay on caller stream.
Capture is allowed only in excluded initialization; seal makes misses eager.
"""
import torch


def eligible(module, x, residual, addition=None, quant_linear=None):
    return (residual is not None and addition is None and quant_linear is None
        and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
        and x.shape[1] == 5120 and 1 <= x.shape[0] <= 64
        and residual.device == x.device and residual.dtype == x.dtype
        and residual.shape == x.shape and x.is_contiguous() and residual.is_contiguous()
        and module.weight.device == x.device and module.weight.dtype == x.dtype
        and module.weight.is_contiguous() and module.hidden_size == 5120
        and not module.fp32_residual and not module.cast_x_before_out_mul
        and module.override_orig_dtype is None and module.variance_size_override is None
        and not torch.cuda.is_current_stream_capturing())


def signature(module, x, residual):
    weight = module.weight
    return (tuple(x.shape), tuple(x.stride()), tuple(residual.stride()),
            x.device.index, x.dtype, torch.cuda.current_stream(x.device).cuda_stream,
            weight.data_ptr(), weight._version, float(module.variance_epsilon))


class Catalog:
    def __init__(self, original):
        self.original = original
        self.entries = {}
        self.sealed = False
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    def execute(self, module, x, residual, addition=None, quant_linear=None):
        if not eligible(module, x, residual, addition, quant_linear):
            self.fallbacks += 1
            return self.original(module, x, residual, addition, quant_linear)
        # Modules are strongly held by entries, avoiding id-reuse collisions.
        key = (module, signature(module, x, residual))
        entry = self.entries.get(key)
        if entry is None:
            if self.sealed:
                self.fallbacks += 1
                return self.original(module, x, residual, addition, quant_linear)
            caller = torch.cuda.current_stream(x.device)
            side = torch.cuda.Stream(device=x.device)
            static_x, static_r = torch.empty_like(x), torch.empty_like(residual)
            static_x.copy_(x)
            static_r.copy_(residual)
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.original(module, static_x, static_r)
            side.synchronize()  # excluded initialization, never steady replay
            graph = torch.cuda.CUDAGraph()
            # thread_local permits unrelated asynchronous output-copy threads;
            # this pure-device callable itself contains no synchronization.
            with torch.cuda.graph(graph, stream=side, capture_error_mode='thread_local'):
                output = self.original(module, static_x, static_r)
            caller.wait_stream(side)
            entry = (graph, static_x, static_r, output, side)
            self.entries[key] = entry
            self.captures += 1
        graph, static_x, static_r, output, _side = entry
        static_x.copy_(x)
        static_r.copy_(residual)
        graph.replay()
        self.replays += 1
        return output

    def summary(self):
        return {'captures': self.captures, 'replays': self.replays,
                'fallbacks': self.fallbacks, 'sealed': self.sealed,
                'rows': sorted({key[1][0][0] for key in self.entries}),
                'entries': len(self.entries)}
