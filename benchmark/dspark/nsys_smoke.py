"""Collector smoke test only; no inference imports and no performance claim."""
import json
import subprocess

import torch

pids = subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
    text=True).strip()
if pids:
    raise RuntimeError('GPU occupied; refusing to run smoke alongside existing work: ' + pids)

a = torch.ones((128, 128), dtype=torch.float32, device='cuda')
b = torch.ones_like(a)
c = torch.empty_like(a)
for _ in range(4):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
assert torch.cuda.cudart().cudaProfilerStart() == 0
for ordinal in range(16):
    with torch.cuda.nvtx.range('nsys_smoke_mm_' + str(ordinal)):
        torch.mm(a, b, out=c)
torch.cuda.synchronize()
assert torch.cuda.cudart().cudaProfilerStop() == 0
assert bool(torch.all(c.cpu() == 128))
print(json.dumps({'status': 'smoke_output_correct', 'expected_nvtx_ranges': 16,
    'expected_mm_launches': 16, 'torch': torch.__version__,
    'gpu': torch.cuda.get_device_name(), 'performance_claim': 'none'}), flush=True)
