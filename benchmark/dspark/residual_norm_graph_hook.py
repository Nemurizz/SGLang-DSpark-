"""Default-off original-arithmetic graph catalog, process-local only."""
import functools
import json
import os
from pathlib import Path


def make_hook(config):
    diagnostic = config.get('graph_diagnostic', False)
    if diagnostic:
        from tpot_trace_hook import make_hook as policy
    else:
        from fixed_fa4_policy_hook import make_hook as policy
    attestation = policy(config)
    import torch
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from residual_norm_graph import Catalog, eligible

    active = os.environ.get('SGLANG_DSPARK_ORIGINAL_NORM_GRAPH') == '1'
    if not diagnostic and not active:
        raise ValueError('Explicit original RMSNorm graph opt-in required')
    if not is_batch_invariant_mode_enabled():
        raise ValueError('Original batch-invariant operators required')
    if getattr(RMSNorm, '_dspark_norm_experiment', False):
        raise ValueError('Do not stack with the rejected norm-fusion experiment')
    if getattr(RMSNorm, '_dspark_original_norm_graph', False):
        raise ValueError('Catalog already installed')
    original = RMSNorm.forward_native
    catalog = Catalog(original)

    @functools.wraps(original)
    def forward(module, x, residual=None, post_residual_addition=None, quant_linear=None):
        supported = eligible(module, x, residual, post_residual_addition, quant_linear)
        if diagnostic:
            torch.cuda.nvtx.range_push('DSparkOriginalNormGraph ' + json.dumps({
                'shape': list(x.shape), 'residual': residual is not None,
                'supported': supported, 'candidate': active and supported}, separators=(',', ':')))
        try:
            return catalog.execute(module, x, residual, post_residual_addition, quant_linear) if active else original(
                module, x, residual, post_residual_addition, quant_linear)
        finally:
            if diagnostic:
                torch.cuda.nvtx.range_pop()

    RMSNorm.forward_native = forward
    RMSNorm._dspark_original_norm_graph = True
    folder = Path(config['graph_policy_folder'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'graph_policy.json').write_text(json.dumps({
        'enabled': active, 'diagnostic': diagnostic, 'process_local': True,
        'original_arithmetic': True, 'private_pools': True,
        'scope': 'BF16 residual RMSNorm H5120 contiguous M1..64 exact shape, per module/stream',
        'framework_graphs': 'OFF', 'attention_kv_planner_sampling_gemm': 'UNCHANGED',
        'ragged_exact_geometry_guard': 'UNCHANGED', 'performance': 'UNPROVEN'}, indent=2))

    def combined(module, inputs, output):
        result = attestation(module, inputs, output)
        if active and not catalog.sealed and (folder / 'seal_catalog').exists():
            catalog.sealed = True
            with (folder / 'catalog_sealed.json').open('x') as handle:
                json.dump(catalog.summary(), handle, indent=2)
        if active and catalog.sealed and (folder / 'snapshot_catalog').exists() and not (folder / 'catalog_final.json').exists():
            with (folder / 'catalog_final.json').open('x') as handle:
                json.dump(catalog.summary(), handle, indent=2)
        return result
    return combined
