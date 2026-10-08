"""SM120 native CUTLASS availability/numerical smoke, NOT a speed benchmark.

Exercise Qwen3-14B linear dimensions and duplicated rows across M. This is not
a model correctness/quality/batch-invariance certificate.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    import torch
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    if tuple(torch.cuda.get_device_capability()) != (12, 0):
        raise ValueError('This preflight is qualified only for SM120')
    if a.output.exists():
        raise ValueError('Refusing to overwrite prior evidence')
    report = {'scope': __doc__, 'torch': torch.__version__,
              'flashinfer': importlib.metadata.version('flashinfer-python'),
              'gpu': torch.cuda.get_device_name(), 'capability': [12, 0],
              'backend': 'cutlass', 'cases': [], 'status': 'running'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261008)
    try:
        for label, n, k in [('qkv', 7168, 5120), ('o_proj', 5120, 5120),
                            ('gate_up', 34816, 5120), ('down', 5120, 17408)]:
            w = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
            x0 = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
            ws = ((448 * 6) / w.float().abs().max()).reshape(1)
            xs = ((448 * 6) / x0.float().abs().max()).reshape(1)
            wq, sfw = nvfp4_quantize(w, ws, sfLayout=SfLayout.layout_128x4,
                                   do_shuffle=False, backend='cuda')
            reference = None
            for m in (1, 8, 32, 64):
                x = x0.repeat(m, 1)
                xq, sfx = nvfp4_quantize(x, xs, sfLayout=SfLayout.layout_128x4,
                                       do_shuffle=False, backend='cuda')
                out = mm_fp4(xq, wq.T, sfx, sfw.T, 1.0 / (xs * ws),
                             torch.bfloat16, backend='cutlass')
                torch.cuda.synchronize()
                if out.shape != (m, n) or not torch.isfinite(out).all().item():
                    raise ValueError('Invalid native NVFP4 output')
                if reference is None:
                    reference = out[0].clone()
                equal = torch.equal(out, reference.unsqueeze(0).expand(m, n))
                report['cases'].append({'linear': label, 'm': m, 'n': n, 'k': k,
                    'finite': True, 'duplicate_rows_exact': equal,
                    'max_abs_delta_from_m1': (out.float() - reference.float()).abs().max().item()})
                print(json.dumps(report['cases'][-1]), flush=True)
                if not equal:
                    raise ValueError('Native NVFP4 row output changed with batch geometry')
            del w, x0, wq, sfw, out, reference
        report['status'] = 'passed_backend_smoke_only'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
