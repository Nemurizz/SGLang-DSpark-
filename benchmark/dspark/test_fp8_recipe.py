import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from fp8_recipe import command, validate_info
from fp8_profile_hook import validate_model


class FP8Tests(unittest.TestCase):
    def test_no_replacement_gemm_or_quantized_draft(self):
        cmd=['python','-m','sglang.launch_server','--attention-backend','flashinfer',
             '--decode-attention-backend','fa4','--speculative-algorithm','DSPARK',
             '--speculative-num-draft-tokens','8','--speculative-draft-attention-backend','flashinfer',
             '--disable-cuda-graph']
        cmd, env=command(cmd, {'PYTHONPATH':'/source'}, SimpleNamespace(), ordinary=False, folder=Path('/evidence'))
        self.assertEqual(cmd[cmd.index('--quantization')+1], 'w8a8_fp8')
        self.assertEqual(cmd[cmd.index('--speculative-draft-model-quantization')+1], 'unquant')
        self.assertEqual(cmd[cmd.index('--kv-cache-dtype')+1], 'bfloat16')
        self.assertEqual(env['USE_TRITON_W8A8_FP8_KERNEL'], '0')
        self.assertEqual(env['SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE'], '0')
        self.assertNotIn('--fp4-gemm-backend',cmd)
        hooks=json.loads(cmd[cmd.index('--forward-hooks')+1])
        self.assertEqual(hooks[1]['hook_factory'], 'fp8_profile_hook:make_model_hook')

    def test_native_fp8_storage_fail_closed(self):
        row=dict(method='W8A8Fp8LinearMethod', weight_dtype='torch.float8_e4m3fn',
                 weight_shape=[5120,7168], scale_dtype='torch.float32',scale_shape=[1,7168],
                 cutlass_supported=True, checkpoint_fp8=False,dynamic_input_scale=True)
        data=dict(kind='target',quantized_linear_count=160,native_cutlass_path=True,
                  native_function='sgl_kernel.gemm.fp8_scaled_mm',linears=[row]*160,
                  embedding_dtype='torch.bfloat16',lm_head_dtype='torch.bfloat16')
        validate_model(data)
        for key, value in [('native_cutlass_path',False),('lm_head_dtype','torch.float8_e4m3fn'),('quantized_linear_count',0)]:
            with self.assertRaises(ValueError):
                validate_model({**data,key:value})
        with self.assertRaises(ValueError):
            validate_model({**data,'linears':[{**row,'scale_shape':[1]}]*160})


if __name__ == '__main__':
    unittest.main()
