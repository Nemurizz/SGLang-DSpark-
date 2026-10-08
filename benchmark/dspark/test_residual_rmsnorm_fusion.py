"""GPU bit-parity gate, not a performance claim. Stop at first mismatch."""
import json
from types import SimpleNamespace
import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim
from residual_rmsnorm_fusion import fused

torch.manual_seed(48)
device = 'cuda'
results = []
with torch.inference_mode():
    for rows in (1, 4, 8, 32, 64):
        for scale in (0.0, 1e-20, 1e-3, 1.0, 1000.0):
            x = (torch.randn(rows, 5120, device=device) * scale).bfloat16()
            r = (torch.randn_like(x.float()) * scale).bfloat16()
            weight = torch.randn(5120, device=device).bfloat16()
            m = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
            summed = x.float() + r.float()
            mean = lambda q: mean_dim(q, -1, keepdim=True)
            expected = ((summed * torch.rsqrt(mean(summed.pow(2)) + m.variance_epsilon)) * weight).bfloat16()
            got, residual = fused(m, x, r, mean_fn=mean)
            output_diff = int((got.view(torch.int16) != expected.view(torch.int16)).sum().item())
            residual_diff = int((residual.view(torch.int16) != summed.bfloat16().view(torch.int16)).sum().item())
            row = {'rows': rows, 'scale': scale, 'output_bit_diffs': output_diff,
                   'residual_bit_diffs': residual_diff}
            results.append(row)
            print(json.dumps(row), flush=True)
            if output_diff or residual_diff:
                raise SystemExit('BIT PARITY FAILED: no service timing may start')
print('PASS', len(results), 'GPU bit-parity cases; not end-to-end correctness/performance proof')
