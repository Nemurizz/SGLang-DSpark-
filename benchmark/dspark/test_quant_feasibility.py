import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from quant_feasibility_gate import quant_command, validate_info, launch_args_from_template
from nvfp4_profile_hook import validate_model, validate_target_linear
from prepare_nvfp4 import OFFICIAL_LFS, REVISION


class QuantGateTests(unittest.TestCase):
    def test_launch_template_scoped_nested_arguments_only(self):
        args = {'repo': '/baseline', 'model': '/bf16'}
        self.assertEqual(vars(launch_args_from_template({'args': args})), args)
        self.assertEqual(vars(launch_args_from_template({
            'execution_integrity_template': {'args': args}, 'policy': 'not inherited'})), args)
        for bad in ({}, {'args': []}, {'args': args,
                    'execution_integrity_template': {'args': {**args, 'model': '/different'}}}):
            with self.assertRaises(ValueError):
                launch_args_from_template(bad)

    def command(self, ordinary):
        cmd = ['python', '-m', 'sglang.launch_server', '--attention-backend', 'flashinfer',
               '--decode-attention-backend', 'fa4', '--tokenizer-path', '/quant',
               '--speculative-algorithm', 'DSPARK', '--speculative-num-draft-tokens', '8',
               '--speculative-draft-attention-backend', 'flashinfer', '--disable-cuda-graph']
        args = SimpleNamespace(tokenizer='/baseline/tokenizer')
        return quant_command(cmd, {'PYTHONPATH': '/baseline/python'}, args,
                             ordinary=ordinary, folder=Path('/evidence'))

    def test_dspark_recipe_separates_weight_draft_kv_graph_cache(self):
        cmd, env = self.command(False)
        for flag, value in (('--fp4-gemm-backend', 'flashinfer_cutlass'),
                            ('--speculative-draft-model-quantization', 'unquant'),
                            ('--speculative-draft-kv-cache-dtype', 'bfloat16'),
                            ('--kv-cache-dtype', 'bfloat16'),
                            ('--tokenizer-path', '/baseline/tokenizer')):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--disable-cuda-graph', cmd)
        self.assertIn('--disable-radix-cache', cmd)
        self.assertNotIn('--quantization', cmd)
        self.assertTrue(all(env[key] == '0' for key in (
            'SGLANG_DSPARK_PREFIX_CACHE', 'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH',
            'SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH', 'SGLANG_DSPARK_COMPACT_SCAN_CUDA')))
        hooks = json.loads(cmd[cmd.index('--forward-hooks') + 1])
        self.assertEqual(hooks[0]['config']['fixed_fa4_tile'], [64, 64])

    def test_quantized_ar_oracle_has_no_draft_arguments(self):
        cmd, _ = self.command(True)
        self.assertFalse(any(s.startswith('--speculative-') for s in cmd))
        self.assertEqual(cmd.count('--disable-radix-cache'), 1)

    def valid_info(self):
        return dict(enable_deterministic_inference=True, disable_cuda_graph=True,
                    disable_radix_cache=True, dtype='bfloat16', kv_cache_dtype='bfloat16',
                    attention_backend='fa4', prefill_attention_backend='fa4',
                    decode_attention_backend='fa4', bf16_gemm_backend='torch',
                    tp_size=1, page_size=1, quantization='modelopt_fp4',
                    fp4_gemm_runner_backend='flashinfer_cutlass', speculative_algorithm='DSPARK',
                    speculative_draft_model_quantization=None,
                    speculative_draft_kv_cache_dtype='bfloat16')

    def test_runtime_flags_fail_closed(self):
        validate_info(self.valid_info(), False)
        for key, wrong in [('kv_cache_dtype', 'fp8_e4m3'), ('quantization', None),
                           ('disable_cuda_graph', False), ('fp4_gemm_runner_backend', 'auto'),
                           ('speculative_draft_model_quantization', 'modelopt_fp4')]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_info({**self.valid_info(), key: wrong}, False)

    def test_runtime_actual_weights_fail_closed(self):
        linear = dict(weight_dtype='torch.uint8', output_dtype='torch.bfloat16',
                      scales=dict(alpha=dict(dtype='torch.float32', numel=1),
                                  input_scale_inv=dict(dtype='torch.float32', numel=1),
                                  weight_scale_interleaved=dict(dtype='torch.float8_e4m3fn', numel=128)))
        validate_target_linear(linear)
        with self.assertRaises(ValueError):
            validate_target_linear({**linear, 'scales': {**linear['scales'],
                                   'input_scale_inv': dict(dtype='torch.bfloat16', numel=1)}})
        target = dict(kind='target', quantized_linear_count=160,
                      quantized_methods=['ModelOptFp4LinearMethod'],
                      fp4_backend='flashinfer_cutlass', embedding_dtype='torch.bfloat16',
                      lm_head_dtype='torch.bfloat16', linears=[linear]*160)
        validate_model(target)
        for key, wrong in [('quantized_linear_count', 0), ('lm_head_dtype', 'torch.uint8'),
                           ('fp4_backend', 'marlin')]:
            with self.assertRaises(ValueError):
                validate_model({**target, key: wrong})
        validate_model({**target, 'kind': 'draft', 'quantized_linear_count': 0,
                        'quantized_methods': [], 'unquantized_linear_count': 20,
                        'weight_dtypes': ['torch.bfloat16']})
        with self.assertRaises(ValueError):
            validate_model({**target, 'kind': 'draft'})

    def test_download_identity_is_frozen(self):
        self.assertEqual(len(REVISION), 40)
        self.assertEqual(len(OFFICIAL_LFS), 3)
        self.assertTrue(all(len(s) == 64 for s in OFFICIAL_LFS.values()))


if __name__ == '__main__':
    unittest.main()
