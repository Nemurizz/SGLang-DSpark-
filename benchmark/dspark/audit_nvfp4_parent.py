"""Read-only checkpoint parent/format gate, including unquantized tensor bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ARCH_FIELDS = ('model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
               'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size',
               'rope_theta', 'rms_norm_eps', 'tie_word_embeddings')


def tensor_info(root, name):
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    path = root / index['weight_map'][name]
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if not 1 <= size <= 32 * 1024 * 1024:
            raise ValueError('Invalid safetensors header')
        header = json.loads(stream.read(size))
    return path, 8 + size, header[name]


def tensor_sha(root, name):
    path, start, info = tensor_info(root, name)
    a, b = info['data_offsets']
    if a < 0 or b <= a or start + b > path.stat().st_size:
        raise ValueError('Invalid tensor byte bounds')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(start + a)
        left = b - a
        while left:
            data = stream.read(min(left, 16 * 1024 * 1024))
            if not data:
                raise ValueError('Truncated tensor')
            h.update(data)
            left -= len(data)
    return {'dtype': info['dtype'], 'shape': info['shape'], 'bytes': b - a, 'sha256': h.hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--quant', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite evidence')
    old = json.loads((a.baseline / 'config.json').read_text())
    new = json.loads((a.quant / 'config.json').read_text())
    if any(old.get(k) != new.get(k) for k in ARCH_FIELDS):
        raise ValueError('Quantized checkpoint does not preserve baseline architecture')
    report = {'status': 'running', 'architecture': {k: new[k] for k in ARCH_FIELDS}, 'tensors': {}}
    try:
        for name in ('model.embed_tokens.weight', 'lm_head.weight'):
            left, right = tensor_sha(a.baseline, name), tensor_sha(a.quant, name)
            if left != right or right['dtype'] != 'BF16':
                raise ValueError('Unquantized tensor changed: ' + name)
            report['tensors'][name] = right
            print('IDENTICAL BF16 ' + name, flush=True)
        # Header-only format checks on representative target transformer weights.
        for suffix in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                       'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
            _, _, info = tensor_info(a.quant, 'model.layers.0.' + suffix + '.weight')
            if info['dtype'] != 'U8':
                raise ValueError('Expected packed uint8 NVFP4: ' + suffix)
        report['status'] = 'passed_parent_and_format_only'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
