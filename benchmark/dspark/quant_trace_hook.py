"""Explicit bounded CUDA/NVTX diagnostic; never installed in quality/timing.

Wrap installed native operators without changing inputs, outputs, tactics or
scales. Files arm/disarm capture; no tensors copied or inspected on the CPU.
"""
import functools
import json
from pathlib import Path


def make_hook(config):
    from nvfp4_profile_hook import make_hook as attestation_factory
    attestation = attestation_factory(config)
    import torch
    import sglang.srt.layers.quantization.modelopt_quant as operators
    folder = Path(config['trace_folder'])
    state = {'active': False, 'closed': False}
    counts = {}

    def update():
        if state['active'] and (folder / 'trace_stop').exists():
            torch.cuda.profiler.stop()
            state.update(active=False, closed=True)
            with (folder / 'trace_counts.json').open('x') as out:
                json.dump(counts, out, indent=2)
        elif not state['active'] and not state['closed'] and (folder / 'trace_start').exists():
            torch.cuda.profiler.start()
            state['active'] = True

    for name in ('fp4_quantize', 'fp4_gemm'):
        original = getattr(operators, name)
        def wrap(original, name):
            @functools.wraps(original)
            def call(*args, **kwargs):
                update()
                if not state['active']:
                    return original(*args, **kwargs)
                label = name + ':' + str(tuple(args[0].shape))
                counts[label] = counts.get(label, 0) + 1
                with torch.cuda.nvtx.range(label):
                    return original(*args, **kwargs)
            return call
        setattr(operators, name, wrap(original, name))
    return attestation
