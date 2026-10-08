"""Existing native per-channel FP8 backend smoke; not quality or E2E."""
import argparse
import json
import os
from pathlib import Path


def run(output):
    if output.exists():
        raise ValueError('Fresh smoke output required')
    os.environ['USE_TRITON_W8A8_FP8_KERNEL'] = '0'
    os.environ['SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE'] = '0'
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, cutlass_fp8_supported
    if torch.cuda.get_device_capability() != (12, 0) or not cutlass_fp8_supported():
        raise ValueError('This recipe requires native SM120 CUTLASS FP8')
    report = {'status': 'running', 'recipe': 'w8a8_fp8_native', 'cases': [],
              'scope': 'Native availability/row invariance, NOT model quality or E2E'}
    torch.manual_seed(20261009)
    try:
        for label, n, k in [('qkv',7168,5120), ('o_proj',5120,5120), ('gate_up',34816,5120), ('down',5120,17408)]:
            w = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
            wq, scales = per_token_group_quant_fp8(w, k)
            wq, scales = wq.t(), scales.t().contiguous()
            x0 = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
            ref = None
            for m in (1,8,32,64):
                out = apply_fp8_linear(x0.repeat(m,1), wq, scales, cutlass_fp8_supported=True)
                torch.cuda.synchronize()
                if out.shape != (m,n) or not torch.isfinite(out).all().item():
                    raise ValueError('Invalid native FP8 output')
                if ref is None:
                    ref = out[0].clone()
                equal = torch.equal(out, ref.unsqueeze(0).expand(m,n))
                report['cases'].append({'linear':label, 'm':m, 'n':n, 'k':k, 'finite':True,
                    'duplicate_rows_exact':equal, 'max_abs_delta':(out.float()-ref.float()).abs().max().item()})
                print(json.dumps(report['cases'][-1]), flush=True)
                if not equal:
                    raise ValueError('Native FP8 row output changes with M')
            del w, wq, scales, x0, out, ref
        report['status'] = 'passed_backend_smoke_only'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
