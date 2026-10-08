"""Experimental non-GEMM pointwise fusion; original batch-invariant mean retained."""
import torch
import triton
import triton.language as tl


@triton.jit
def _residual_square_stage(X, R, S, Q, RO, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < N
    value = tl.load(X + i, active, 0).to(tl.float32)
    value += tl.load(R + i, active, 0).to(tl.float32)
    tl.store(S + i, value, active)
    tl.store(Q + i, value * value, active)
    tl.store(RO + i, value.to(tl.bfloat16), active)


@triton.jit
def _norm_output_stage(S, V, W, Y, N: tl.constexpr, H: tl.constexpr,
                       EPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < N
    variance = tl.load(V + i // H, active, 0)
    inv = tl.rsqrt(variance + EPS)
    value = tl.load(S + i, active, 0) * inv
    value *= tl.load(W + i % H, active, 0).to(tl.float32)
    tl.store(Y + i, value.to(tl.bfloat16), active)


def eligible(module, x, residual, post=None, quant_linear=None):
    return (residual is not None and post is None and quant_linear is None
            and x.is_cuda and x.dtype == residual.dtype == module.weight.dtype == torch.bfloat16
            and x.device == residual.device == module.weight.device
            and x.ndim == 2 and x.shape == residual.shape and x.shape[-1] == 5120
            and 0 < x.shape[0] <= 64 and module.hidden_size == 5120
            and x.is_contiguous() and residual.is_contiguous() and module.weight.is_contiguous()
            and tuple(module.weight.shape) == (5120,)
            and module.override_orig_dtype is None and module.variance_size_override is None
            and not module.cast_x_before_out_mul and not module.fp32_residual
            and not x.requires_grad and not residual.requires_grad
            and not torch.cuda.is_current_stream_capturing())


def fused(module, x, residual, *, mean_fn=None):
    """No modified reduction, dtype, GEMM, KV, sampler, or acceptance policy."""
    summed = torch.empty_like(x, dtype=torch.float32)
    squared = torch.empty_like(summed)
    residual_out, output = torch.empty_like(x), torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 256),)
    _residual_square_stage[grid](x, residual, summed, squared, residual_out,
                                x.numel(), 256, enable_fp_fusion=False)
    variance = squared.mean(dim=-1, keepdim=True) if mean_fn is None else mean_fn(squared)
    _norm_output_stage[grid](summed, variance, module.weight, output, x.numel(),
                             x.shape[-1], module.variance_epsilon, 256,
                             enable_fp_fusion=False)
    return output, residual_out
