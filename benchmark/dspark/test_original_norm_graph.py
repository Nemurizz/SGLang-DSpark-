"""GPU parity/lifetime/miss/shape gates, not a speed benchmark."""
import json
import torch
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
from sglang.srt.layers.layernorm import RMSNorm
from residual_norm_graph import Catalog


def run():
    enable_batch_invariant_mode()
    torch.manual_seed(73)
    module = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    module.weight.data.copy_(torch.randn_like(module.weight))
    original = RMSNorm.forward_native
    catalog = Catalog(original)
    cases = 0
    with torch.inference_mode():
        for rows in (1, 4, 8, 16, 32, 64):
            for scale in (0.0, 1e-20, 1e-3, 1.0, 1000.0):
                x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16) * scale
                r = torch.randn_like(x) * scale
                expected = original(module, x, r)
                x_before, r_before = x.clone(), r.clone()
                got = catalog.execute(module, x, r)
                assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a,b in zip(expected,got))
                assert torch.equal(x, x_before) and torch.equal(r, r_before)
                # Consumer before reuse must finish before overwritten persistent output.
                saved = tuple(value.clone() for value in got)
                catalog.execute(module, torch.zeros_like(x), torch.zeros_like(r))
                assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a,b in zip(expected,saved))
                cases += 1
        catalog.sealed = True
        before = catalog.captures
        x = torch.randn(3, 5120, device='cuda', dtype=torch.bfloat16)
        r = torch.randn_like(x)
        expected, got = original(module,x,r), catalog.execute(module,x,r)
        assert all(torch.equal(a,b) for a,b in zip(expected,got)) and catalog.captures == before
        # A separate caller stream uses separate entries; sealed means eager.
        with torch.cuda.stream(torch.cuda.Stream()):
            expected, got = original(module,x,r), catalog.execute(module,x,r)
            assert all(torch.equal(a,b) for a,b in zip(expected,got))
        torch.cuda.synchronize()
    print(json.dumps({'passed': True, 'parity_cases': cases, 'summary': catalog.summary()}))


if __name__ == '__main__':
    run()
