"""Existing SGLang native W8A8 FP8, loading unchanged parent BF16 checkpoint."""
import json
from pathlib import Path

from stable_baseline_gate import profile_command

RECIPE = 'w8a8_fp8_native'
PROFILE = 'dspark-qwen3-14b-online-w8a8-fp8-native-bf16-draft-fa4fixed64-h66-v1'


def command(cmd, env, args, *, ordinary, folder):
    cmd, env = profile_command(cmd, env, ordinary=ordinary, profile_id=PROFILE,
        runtime_folder=folder / 'runtime', hook_folder=Path(__file__).resolve().parent,
        fixed_tile=[64, 64])
    cmd += ['--quantization', 'w8a8_fp8']
    if '--disable-radix-cache' not in cmd:
        cmd.append('--disable-radix-cache')
    if not ordinary:
        cmd += ['--speculative-draft-model-quantization', 'unquant',
                '--speculative-draft-kv-cache-dtype', 'bfloat16']
    env['USE_TRITON_W8A8_FP8_KERNEL'] = '0'
    env['SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE'] = '0'
    for key in ('SGLANG_DSPARK_PREFIX_CACHE', 'SGLANG_DSPARK_PREFIX_CACHE_SHADOW',
                'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH', 'SGLANG_DSPARK_COMPACT_SCAN_CUDA',
                'SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH'):
        env[key] = '0'
    hooks = json.loads(cmd[cmd.index('--forward-hooks') + 1])
    hooks[0]['hook_factory'] = 'fp8_profile_hook:make_hook'
    hooks.append({'name': 'fp8-storage-attestation', 'target_modules': ['model'],
        'hook_factory': 'fp8_profile_hook:make_model_hook', 'config': {'folder': str(folder / 'runtime')}})
    if not ordinary:
        hooks.append({'name': 'bf16-draft-storage-attestation',
            'target_modules': [f'layers.{i}' for i in range(5)],
            'hook_factory': 'fp8_profile_hook:make_draft_hook', 'config': {'folder': str(folder / 'runtime')}})
    cmd[cmd.index('--forward-hooks') + 1] = json.dumps(hooks)
    return cmd, env


def validate_info(info, ordinary):
    expected = dict(enable_deterministic_inference=True, disable_cuda_graph=True,
        disable_radix_cache=True, dtype='bfloat16', kv_cache_dtype='bfloat16',
        attention_backend='fa4', prefill_attention_backend='fa4', decode_attention_backend='fa4',
        bf16_gemm_backend='torch', tp_size=1, page_size=1, quantization='w8a8_fp8')
    if not ordinary:
        expected.update(speculative_algorithm='DSPARK', speculative_draft_model_quantization=None,
                        speculative_draft_kv_cache_dtype='bfloat16')
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f'Unexpected W8A8 FP8 runtime {key}: {info.get(key)!r}')
