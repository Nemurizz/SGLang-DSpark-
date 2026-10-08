"""Read-only attestation for native online per-channel/per-token W8A8 FP8.

No custom GEMM, native operator substitution, tensor copies, or timing probes.
"""
import json
from pathlib import Path

from nvfp4_profile_hook import make_hook, make_draft_hook


def validate_model(data):
    if data['kind'] == 'draft':
        from nvfp4_profile_hook import validate_model as validate_draft
        return validate_draft(data)
    if data['kind'] != 'target' or data['quantized_linear_count'] != 160:
        raise ValueError('Expected 160 native W8A8 FP8 transformer linears')
    if not data['native_cutlass_path'] or data['native_function'] != 'sgl_kernel.gemm.fp8_scaled_mm':
        raise ValueError('Native CUTLASS FP8 path not attested')
    if len(data['linears']) != 160:
        raise ValueError('Missing per-layer storage evidence')
    for row in data['linears']:
        k, n = row['weight_shape']
        if (row['method'] != 'W8A8Fp8LinearMethod' or row['weight_dtype'] != 'torch.float8_e4m3fn'
                or row['scale_dtype'] != 'torch.float32' or row['scale_shape'] != [1, n]
                or not row['cutlass_supported'] or row['checkpoint_fp8'] or not row['dynamic_input_scale']
                or k % 16 or n % 16):
            raise ValueError('Unexpected FP8 granularity/backend/storage: ' + repr(row))
    if data['embedding_dtype'] != 'torch.bfloat16' or data['lm_head_dtype'] != 'torch.bfloat16':
        raise ValueError('Shared target head/embed must stay BF16')


def make_model_hook(config):
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.environ import envs
    if fp8_utils.use_triton_w8a8_fp8_kernel or envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get():
        raise ValueError('This experiment forbids Triton replacement/tuned GEMM')
    native = fp8_utils.fp8_scaled_mm
    recorded = False
    folder = Path(config['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    def hook(module, inputs, output):
        nonlocal recorded
        if recorded:
            return None
        if type(module).__name__ != 'Qwen3Model':
            raise ValueError('Wrong target backbone')
        linears = []
        for name, layer in module.named_modules():
            method = getattr(layer, 'quant_method', None)
            if type(method).__name__ == 'W8A8Fp8LinearMethod':
                linears.append({'name': name, 'method': type(method).__name__,
                    'weight_dtype': str(layer.weight.dtype), 'weight_shape': list(layer.weight.shape),
                    'scale_dtype': str(layer.weight_scale.dtype), 'scale_shape': list(layer.weight_scale.shape),
                    'cutlass_supported': method.cutlass_fp8_supported,
                    'checkpoint_fp8': method.quantization_config.is_checkpoint_fp8_serialized,
                    'dynamic_input_scale': layer.input_scale is None})
        data = {'kind': 'target', 'quantized_linear_count': len(linears), 'linears': linears,
            'native_cutlass_path': not fp8_utils.use_triton_w8a8_fp8_kernel and not envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get(),
            'native_function': native.__module__ + '.' + native.__name__,
            'embedding_dtype': str(module.embed_tokens.weight.dtype),
            'scope': 'native dispatch configuration and actual loaded weights, NOT quality'}
        with (folder / 'target-backbone.json').open('x') as out:
            json.dump(data, out, indent=2)
        recorded = True
        return None
    return hook
