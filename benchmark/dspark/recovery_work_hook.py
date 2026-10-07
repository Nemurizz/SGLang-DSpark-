"""Default-off worker work observer for correctness runs, NEVER timing A/B.

Wrap existing functions without changing metadata, tensors or scheduling. Record
real prefill geometry and successful draft injection enqueue counts. Completion
and token checks in the driver establish that asynchronous writes succeeded.
"""
import contextvars
import json
import os
from pathlib import Path
import time


def validate_work(record):
    if record['actual_prefill_tokens'] != record['draft_injected_tokens']:
        raise ValueError('Prefill/injection token count mismatch')
    if record['prompt_tokens'] != record['actual_prefix_tokens'] + record['actual_prefill_tokens']:
        raise ValueError('Unexpected chunked/mixed prefill geometry')
    if record['plan_mode'] in ('partial', 'full') and record['target_hit_tokens'] > 0:
        if record['actual_prefix_tokens'] != record['paired_hit_tokens']:
            raise ValueError('Protected prefix differs from paired readiness')
        if record['actual_recovered_tokens'] != record['planned_recovery_tokens']:
            raise ValueError('Actual suffix work differs from recovery plan')


def make_hook(config):
    if os.environ.get('SGLANG_DSPARK_RECOVERY_WORK_OBSERVER') != '1':
        raise ValueError('Recovery observer requires explicit correctness-only opt-in')
    from fixed_fa4_policy_hook import make_hook as numerical_hook
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
    from sglang.srt.speculative.dspark_components.dspark_kv_inject import TargetHiddenKvInjector

    marker = getattr(DSparkWorkerV2, '_dspark_recovery_observer', None)
    path = str(Path(config['work_folder']) / 'worker_work.jsonl')
    if marker is not None:
        if marker != (config['profile_id'], path):
            raise ValueError('Conflicting recovery observer')
        return numerical_hook(config)
    Path(config['work_folder']).mkdir(parents=True, exist_ok=True)
    active = contextvars.ContextVar('dspark_prefill_observation', default=None)
    original_forward = DSparkWorkerV2._forward_prefill
    original_inject = TargetHiddenKvInjector.inject_target_hidden

    def inject(self, *args, **kwargs):
        context = active.get()
        if context is not None:
            if args or kwargs.get('cache_loc') is None or kwargs.get('target_hidden') is None:
                raise ValueError('Unknown prefill injection call contract')
            n = kwargs['cache_loc'].numel()
            if kwargs['target_hidden'].shape[0] < n or kwargs['positions'].numel() != n:
                raise ValueError('Insufficient captured hidden states')
        result = original_inject(self, *args, **kwargs)
        if context is not None:
            context['successful_injection_calls'] += 1
            context['injected_tokens'] += n
        return result

    def forward(self, batch, *args, **kwargs):
        if batch.forward_mode.is_idle():
            return original_forward(self, batch, *args, **kwargs)
        context = {'successful_injection_calls': 0, 'injected_tokens': 0}
        token = active.set(context)
        try:
            result = original_forward(self, batch, *args, **kwargs)
            if context != {'successful_injection_calls': 1, 'injected_tokens': sum(batch.extend_lens)}:
                raise ValueError('Unexpected successful prefill injection geometry')
            with open(path, 'a', encoding='utf-8') as stream:
                for req, prefix, extend in zip(batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True):
                    target = int(req.dspark_target_prefix_len)
                    record = {'rid': str(req.rid), 'wall_time_s': time.time(),
                        'profile_id': config['profile_id'], 'prompt_tokens': len(req.origin_input_ids),
                        'actual_prefix_tokens': int(prefix), 'actual_prefill_tokens': int(extend),
                        'draft_injected_tokens': int(extend), 'target_hit_tokens': target,
                        'paired_hit_tokens': int(req.dspark_paired_prefix_len),
                        'planned_recovery_tokens': int(req.dspark_recovery_tokens),
                        'actual_recovered_tokens': max(0, target - int(prefix)),
                        'plan_mode': req.dspark_recovery_mode,
                        'draft_ready_after_injection': bool(req.dspark_draft_kv_ready),
                        'measurement': 'successful enqueue; driver requires completed token/acceptance match'}
                    validate_work(record)
                    stream.write(json.dumps(record) + '\n')
            return result
        finally:
            active.reset(token)

    DSparkWorkerV2._forward_prefill = forward
    TargetHiddenKvInjector.inject_target_hidden = inject
    DSparkWorkerV2._dspark_recovery_observer = (config['profile_id'], path)
    return numerical_hook(config)
