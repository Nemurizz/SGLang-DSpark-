"""Install process-local policy BEFORE target capture, including spawned workers.

The regular forward-hook factory runs AFTER capture; it cannot configure capture.
Keep this module as multiprocessing's main entry point, not runpy(__main__).
"""
import functools
import json
import os
from pathlib import Path


def install():
    if os.environ.get('SGLANG_DSPARK_EXACT_TARGET_GRAPH8') != '1':
        raise ValueError('This launcher is exclusively an explicit graph experiment')
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as setup
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
    if getattr(setup.capture_decode_graph, '_exact_graph8_bootstrap', False):
        return
    original = setup.capture_decode_graph
    @functools.wraps(original)
    def capture(*, model_runner):
        if model_runner.is_draft_worker:
            raise ValueError('Draft decode capture must remain OFF')
        if not model_runner.spec_algorithm.is_dspark():
            raise ValueError('Unexpected non-DSpark target')
        if not getattr(model_runner.attn_backend, 'supports_ragged_verify_graph', False):
            raise ValueError('Backend capability missing; never override it')
        from sglang.srt.environ import envs
        if not envs.SGLANG_DSPARK_RAGGED_VERIFY_GRAPH.get():
            raise ValueError('Explicit ragged graph opt-in missing')
        configs = [hook['config'] for hook in model_runner.server_args.forward_hooks
                   if hook['hook_factory'].startswith('exact_target_')]
        if len(configs) != 1:
            raise ValueError('Requires one explicit experiment hook config')
        from exact_target_graph_policy import make_hook
        make_hook(configs[0])  # No registration/tensor hook inside captured graph.
        result = original(model_runner=model_runner)
        runner = result.runner
        if runner is None or runner.capture_num_tokens != [8]:
            raise ValueError('No actual target 8-token capture')
        layout = runner._captured_ragged_layouts[8]
        if layout.bs != 1 or layout.verify_lens_cpu != [8]:
            raise ValueError('Capture request-slot geometry must be exactly 1x8')
        folder = Path(configs[0]['graph_policy_folder'])
        with (folder/'capture_actual.json').open('x') as out:
            json.dump({'bs': layout.bs, 'verify_lens_cpu': layout.verify_lens_cpu,
                'tokens': runner.capture_num_tokens, 'buffer_max_bs': runner.max_bs,
                'backend_class': type(model_runner.attn_backend).__name__,
                'capture_seconds': result.capture_time, 'memory_gb': result.memory_usage_gb}, out, indent=2)
        return result
    capture._exact_graph8_bootstrap = True
    setup.capture_decode_graph = capture

    original_draft = DSparkWorkerV2.init_cuda_graphs
    @functools.wraps(original_draft)
    def draft(self):
        # The policy installed above also sets this; assert it after draft init.
        self._decode_graph_allowed = False
        result = original_draft(self)
        configs = [hook['config'] for hook in self.model_runner.server_args.forward_hooks
                   if hook['hook_factory'].startswith('exact_target_')]
        if len(configs) != 1 or self._decode_graph_allowed:
            raise ValueError('Cannot attest draft graph OFF')
        with (Path(configs[0]['graph_policy_folder'])/'draft_actual.json').open('x') as out:
            json.dump({'decode_graph_allowed': self._decode_graph_allowed,
                       'runner_class': type(self.draft_model_runner.decode_cuda_graph_runner).__name__}, out)
        return result
    DSparkWorkerV2.init_cuda_graphs = draft


install()

if __name__ == '__main__':
    import sys
    from sglang.launch_server import load_plugins, prepare_server_args, run_server, kill_process_tree
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
