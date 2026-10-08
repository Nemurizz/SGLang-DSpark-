"""Read-only model attestation plus the existing explicit FA4 tile policy.

NVFP4 is a new arithmetic contract: BF16 batch-invariant settings do NOT prove
the quantized GEMM is batch invariant. The separate token gate must establish it.
No tensor copies, timing instrumentation, asynchronous stack dump, or sitecustomize.
"""
import json
from pathlib import Path


def make_hook(config):
    from fixed_fa4_policy_hook import make_hook as fixed_hook
    attention_hook = fixed_hook(config)
    recorded = False
    def hook(module, inputs, output):
        nonlocal recorded
        result = attention_hook(module, inputs, output)
        if not recorded and len(inputs) >= 4:
            head = inputs[2]
            data = {'lm_head_dtype': str(head.weight.dtype),
                    'lm_head_shape': list(head.weight.shape),
                    'lm_head_method': type(head.quant_method).__name__}
            if data['lm_head_dtype'] != 'torch.bfloat16' or data['lm_head_shape'][-1] != 5120:
                raise ValueError('Shared target LM head must be a dense BF16 5120-wide head')
            with (Path(config['folder']) / 'target-head.json').open('x') as stream:
                json.dump(data, stream, indent=2)
            recorded = True
        return result
    return hook


def validate_target_linear(data):
    if data['weight_dtype'] != 'torch.uint8' or data['output_dtype'] != 'torch.bfloat16':
        raise ValueError('Expected native packed FP4 weights and BF16 linear output')
    for name in ('alpha', 'input_scale_inv'):
        if data['scales'][name] != {'dtype': 'torch.float32', 'numel': 1}:
            raise ValueError('Global FP4 scale must be a float32 scalar: ' + name)
    if data['scales']['weight_scale_interleaved']['dtype'] != 'torch.float8_e4m3fn':
        raise ValueError('Expected FP8 block scales for native NVFP4')


def validate_model(data):
    if data['kind'] == 'target':
        if data['quantized_linear_count'] != 160:
            raise ValueError('Expected all 160 fused target transformer linears to use NVFP4')
        if set(data['quantized_methods']) != {'ModelOptFp4LinearMethod'}:
            raise ValueError('Unexpected target quantization implementation')
        if data['fp4_backend'] != 'flashinfer_cutlass':
            raise ValueError('Wrong FP4 backend')
        if len(data.get('linears', [])) != 160:
            raise ValueError('Missing actual per-linear storage/scale attestation')
        for linear in data['linears']:
            validate_target_linear(linear)
    elif data['kind'] == 'draft':
        if data['quantized_linear_count'] or data.get('unquantized_linear_count') != 20:
            raise ValueError('All 20 fused linears of the five-layer draft must remain unquantized')
        if set(data.get('weight_dtypes', [])) != {'torch.bfloat16'}:
            raise ValueError('All draft linears must remain BF16')
        return
    else:
        raise ValueError('Unknown model kind')
    for key in ('embedding_dtype', 'lm_head_dtype'):
        if data[key] != 'torch.bfloat16':
            raise ValueError('Shared embedding/head must remain BF16: ' + key)


def make_model_hook(config):
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
    folder = Path(config['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    recorded = False

    def hook(module, inputs, output):
        nonlocal recorded
        if recorded:
            return None
        # Top-level Qwen3ForCausalLM.forward is called directly by the runner,
        # bypassing its nn.Module hooks. Hook the actual called Qwen3Model body.
        if type(module).__name__ != 'Qwen3Model':
            raise ValueError('Target backbone hook attached to an unknown class')
        linears = []
        for name, layer in module.named_modules():
            method = getattr(layer, 'quant_method', None)
            if method is not None and type(method).__name__ == 'ModelOptFp4LinearMethod':
                linears.append({'name': name, 'method': type(method).__name__,
                                'weight_dtype': str(layer.weight.dtype),
                                'packed_weight_shape': list(layer.weight.shape),
                                'output_dtype': str(layer.params_dtype),
                                'scales': {key: {'dtype': str(getattr(layer, key).dtype),
                                                 'numel': getattr(layer, key).numel()}
                                           for key in ('alpha', 'input_scale_inv', 'weight_scale_interleaved')}})
        embed = getattr(getattr(module, 'model', None), 'embed_tokens', None)
        if embed is None:
            embed = getattr(module, 'embed_tokens', None)
        data = {'kind': 'target', 'model_class': type(module).__name__,
                'quantized_linear_count': len(linears),
                'quantized_methods': sorted({r['method'] for r in linears}),
                'fp4_backend': get_fp4_gemm_runner_backend().value,
                'embedding_dtype': str(getattr(getattr(embed, 'weight', None), 'dtype', None)),
                'linears': linears,
                'scope': 'actual runtime storage/backend only, NOT numerical correctness'}
        with (folder / 'target-backbone.json').open('x') as stream:
            json.dump(data, stream, indent=2)
        recorded = True
        return None
    return hook


def make_draft_hook(config):
    folder = Path(config['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    layers = {}
    def hook(module, inputs, output):
        if id(module) in layers:
            return None
        linears = [layer for layer in module.modules() if hasattr(layer, 'quant_method')
                   and hasattr(layer, 'weight')]
        if len(linears) != 4 or any(type(layer.quant_method).__name__ != 'UnquantizedLinearMethod'
                                    or str(layer.weight.dtype) != 'torch.bfloat16' for layer in linears):
            raise ValueError('Draft decoder layer must contain exactly four BF16 unquantized linears')
        layers[id(module)] = len(linears)
        if len(layers) == 5:
            data = {'kind': 'draft', 'quantized_linear_count': 0,
                    'unquantized_linear_count': sum(layers.values()),
                    'weight_dtypes': ['torch.bfloat16'],
                    'shared_head_embedding': 'target modules; attach_shared_modules source unchanged'}
            validate_model(data)
            with (folder / 'draft-weights.json').open('x') as stream:
                json.dump(data, stream, indent=2)
        return None
    return hook
