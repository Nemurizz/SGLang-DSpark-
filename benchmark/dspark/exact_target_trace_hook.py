"""Default-off host/NVTX diagnosis. No tensor copies, synchronizes or decisions.

CUDA profiler Start/Stop only delimit the separately labelled nsys window.
Buffer host measurements in memory; flush once before a later excluded request.
Elapsed host spans include waits, thread CPU spans exclude descheduling/waits.
"""
import contextvars
import functools
import json
import os
from pathlib import Path
import time


def make_hook(config):
    if os.environ.get('SGLANG_DSPARK_TPOT_TRACE') != '1':
        raise ValueError('Explicit diagnostic opt-in required')
    from exact_target_graph_policy import make_hook as policy
    attestation = policy(config)
    import torch
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
    from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
    from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    if getattr(DSparkWorkerV2, '_tpot_diagnostic_installed', False):
        return attestation
    folder = Path(config['diagnostic_folder'])
    context = contextvars.ContextVar('tpot_diagnostic_context', default={})
    records, steps = [], {}
    state = {'active': False, 'closed': False}

    def flush():
        with (folder / 'worker_trace.jsonl').open('x') as out:
            for row in records:
                out.write(json.dumps(row) + '\n')

    def wrap(cls, name, kind):
        original = getattr(cls, name)

        @functools.wraps(original)
        def measured(self, *args, **kwargs):
            token = None
            if kind == 'prefill':
                if (folder / 'trace_stop').exists() and state['active']:
                    if config['nsys']:
                        torch.cuda.profiler.stop()
                    state.update(active=False, closed=True)
                    flush()
                if not state['closed'] and not state['active'] and (folder / 'trace_start').exists():
                    if config['nsys']:
                        torch.cuda.profiler.start()
                    state['active'] = True
            if not state['active']:
                return original(self, *args, **kwargs)
            if kind in ('prefill', 'decode'):
                batch = args[0]
                if len(batch.reqs) != 1:
                    raise ValueError('Diagnostic only supports one request')
                req = batch.reqs[0]
                rid = str(req.rid)
                if kind == 'decode':
                    steps[rid] = steps.get(rid, 0) + 1
                token = context.set({'rid': rid, 'step': steps.get(rid, 0),
                    'protected_prefix': req.cache_protected_len,
                    'input_tokens': len(req.origin_input_ids),
                    'tree_cache_class': type(getattr(batch, 'tree_cache', None)).__name__})
            elif name.startswith('cache_') and args:
                req = args[0]
                rid = str(req.rid)
                token = context.set({'rid': rid, 'step': steps.get(rid, 0),
                    'protected_prefix': req.cache_protected_len,
                    'input_tokens': len(req.origin_input_ids), 'tree_cache_class': type(self).__name__})
            details = dict(context.get())
            label = 'DSpark.' + name
            torch.cuda.nvtx.range_push(label + ' ' + json.dumps(details, separators=(',', ':')))
            started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                ended, cpu_ended = time.perf_counter_ns(), time.thread_time_ns()
                torch.cuda.nvtx.range_pop()
                records.append({**details, 'method': name, 'kind': kind,
                    'start_ns': started, 'end_ns': ended, 'wall_ns': ended - started,
                    'thread_cpu_ns': cpu_ended - cpu_started})
                if token is not None:
                    context.reset(token)
        setattr(cls, name, measured)

    wrap(DSparkWorkerV2, '_forward_prefill', 'prefill')
    wrap(DSparkWorkerV2, '_forward_decode', 'decode')
    for cls, names in (
        (DraftBlockProposer, ('propose',)),
        (TargetVerifyExecutor, ('run_compact', 'run_non_compact', 'accept_and_finalize', 'commit_hidden')),
        (DSparkVerifyPlanner, ('resolve_verify_token_budget', 'schedule_layout', 'select_verify_path')),
        (UnifiedRadixCache, ('cache_finished_req', 'cache_unfinished_req')),
    ):
        for name in names:
            wrap(cls, name, 'segment')
    DSparkWorkerV2._tpot_diagnostic_installed = True
    return attestation
