"""Independent opt-in numerical-equivalent target-only 1x8 graph experiment.

Original fixed64 FA4 policy and GEMM retained. Original exact admission guard
called first, unchanged; capture catalog narrowed to [8]. Draft graph OFF.
"""
import functools
import json
import os
from pathlib import Path


def make_hook(config):
    import torch
    from fixed_fa4_policy_hook import TILE, locked_options, validate_empty_compile_cache
    from sglang.kernels.ops.attention.fa4_sm120.runtime import Sm120ForwardHost
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
    from exact_target_runtime import make_hook as attest

    if os.environ.get('SGLANG_DSPARK_EXACT_TARGET_GRAPH8') != '1':
        raise ValueError('Explicit isolated target graph opt-in required')
    if config.get('fixed_fa4_tile') != list(TILE) or torch.cuda.get_device_capability() != (12,0):
        raise ValueError('Unqualified device/tile')
    if not get_exec().deterministic.enable_deterministic_inference or not (
            check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
            and check_cuda_graph_backend(Phase.DECODE, Backend.FULL)):
        raise ValueError('Requires deterministic, prefill OFF, decode full')
    if list(get_exec().graph.cuda_graph_config.decode.bs) != [1,4,8]:
        raise ValueError('Graph buffer capacity must cover concurrent eager fallbacks')
    folder = Path(config['graph_policy_folder'])
    installed = getattr(DecodeCudaGraphRunner, '_dspark_exact_graph8', None)
    if installed:
        if installed['folder'] != str(folder):
            raise ValueError('Conflicting graph experiment')
        return installed['hook']
    validate_empty_compile_cache(_flash_attn_fwd.compile_cache)
    original_select = Sm120ForwardHost.select_config
    tile_state = {'tile': list(TILE), 'profile_id': config['profile_id'], 'select_calls': 0}
    def select_config(**options):
        value = original_select(**locked_options(options))
        if (value.tile_m,value.tile_n) != TILE:
            raise ValueError('Existing fixed64 host policy not honored')
        tile_state['select_calls'] += 1
        return value
    Sm120ForwardHost.select_config = staticmethod(select_config)
    Sm120ForwardHost._dspark_fixed_tile_policy = tile_state

    # Capture only the full-width 1x8 target geometry. Never relax admission.
    DecodeCudaGraphRunner._build_ragged_verify_token_buckets = lambda self: [8]
    original_guard = DecodeCudaGraphRunner._can_run_ragged_verify_graph
    counters = {'admitted': 0, 'fallback': 0, 'admitted_by_bs': {}}
    @functools.wraps(original_guard)
    def guard(self, fb, layout):
        allowed = original_guard(self, fb, layout)
        if allowed:
            if fb.batch_size != 1 or layout.graph_num_tokens != 8 or layout.verify_lens_cpu != [8]:
                raise ValueError('Original exact guard admitted an unexpected geometry')
            counters['admitted'] += 1
            counters['admitted_by_bs'][str(fb.batch_size)] = counters['admitted_by_bs'].get(str(fb.batch_size),0)+1
        else:
            counters['fallback'] += 1
        return allowed
    DecodeCudaGraphRunner._can_run_ragged_verify_graph = guard
    original_draft_init = DSparkWorkerV2.init_cuda_graphs
    @functools.wraps(original_draft_init)
    def draft_init(self):
        self._decode_graph_allowed = False
        return original_draft_init(self)
    DSparkWorkerV2.init_cuda_graphs = draft_init

    folder.mkdir(parents=True, exist_ok=True)
    (folder/'graph_policy.json').write_text(json.dumps({'enabled': True, 'process_local': True,
        'diagnostic': config.get('graph_diagnostic',False), 'gemm': 'UNCHANGED',
        'original_exact_guard': 'CALLED UNCHANGED; additional 1x8 restriction',
        'target_capture_tokens': [8], 'draft_graph': False, 'prefill_graph': False,
        'framework_target_decode_graph': 'full opt-in', 'performance': 'UNPROVEN'},indent=2))
    attestation = attest(config)
    def combined(module, inputs, output):
        result = attestation(module,inputs,output)
        if counters['admitted'] and not (folder/'graph_calls.json').exists():
            with (folder/'graph_calls.json').open('x') as out:json.dump(counters,out,indent=2)
        if (folder/'snapshot_graph').exists() and not (folder/'graph_final.json').exists():
            with (folder/'graph_final.json').open('x') as out:json.dump(counters,out,indent=2)
        return result
    DecodeCudaGraphRunner._dspark_exact_graph8 = {'folder':str(folder), 'hook':combined}
    return combined
