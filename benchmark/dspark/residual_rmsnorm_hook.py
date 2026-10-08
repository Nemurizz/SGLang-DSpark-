"""Default-off process-local kernel experiment and separate NVTX attribution."""
import functools
import json
import os
from pathlib import Path


def make_hook(config):
    diagnostic = config.get('norm_diagnostic', False)
    if diagnostic:
        from tpot_trace_hook import make_hook as policy
    else:
        from fixed_fa4_policy_hook import make_hook as policy
    attestation = policy(config)
    import torch
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from residual_rmsnorm_fusion import eligible, fused

    active = os.environ.get('SGLANG_DSPARK_RESIDUAL_RMSNORM_FUSION') == '1'
    if not diagnostic and not active:
        raise ValueError('Explicit residual RMSNorm opt-in required')
    if not is_batch_invariant_mode_enabled():
        raise ValueError('Original batch-invariant mean must remain installed')
    if getattr(RMSNorm, '_dspark_norm_experiment', False):
        return attestation
    original = RMSNorm.forward_native
    counters = {'eligible_calls': 0, 'fallback_calls': 0}

    @functools.wraps(original)
    def forward(module, x, residual=None, post_residual_addition=None, quant_linear=None):
        supported = eligible(module, x, residual, post_residual_addition, quant_linear)
        selected = active and supported
        counters['eligible_calls' if supported else 'fallback_calls'] += 1
        if diagnostic:
            torch.cuda.nvtx.range_push('DSparkResidualNorm ' + json.dumps({
                'shape': list(x.shape), 'residual': residual is not None,
                'supported': supported, 'candidate': selected}, separators=(',', ':')))
        try:
            return fused(module, x, residual) if selected else original(
                module, x, residual, post_residual_addition, quant_linear)
        finally:
            if diagnostic:
                torch.cuda.nvtx.range_pop()

    RMSNorm.forward_native = forward
    RMSNorm._dspark_norm_experiment = True
    folder = Path(config.get('norm_policy_folder', config['folder']))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'norm_policy.json').write_text(json.dumps({
        'enabled': active, 'diagnostic': diagnostic,
        'scope': 'BF16 residual RMSNorm H5120, contiguous M1..64, eager only',
        'mean': 'UNCHANGED original batch-invariant aten mean',
        'gemm': 'UNCHANGED', 'graphs': 'UNCHANGED OFF',
        'process_local': True, 'performance': 'UNPROVEN'}, indent=2))

    def combined(module, inputs, output):
        result = attestation(module, inputs, output)
        if not (folder / 'norm_calls.json').exists() and counters['eligible_calls']:
            with (folder / 'norm_calls.json').open('x') as handle:
                json.dump(counters, handle)
        return result
    return combined
