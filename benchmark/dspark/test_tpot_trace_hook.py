import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import tpot_trace_hook


class HookTest(unittest.TestCase):
    def modules(self):
        class Worker:
            def _forward_prefill(self, batch, publish):
                return ('prefill', batch)
            def _forward_decode(self, batch, publish):
                return ('decode', batch)
        class Segment:
            def original(self, *args, **kwargs):
                return kwargs.get('value', 'untouched')
        for method in ('propose', 'run_compact', 'run_non_compact', 'accept_and_finalize',
                       'commit_hidden', 'resolve_verify_token_budget', 'schedule_layout',
                       'select_verify_path', 'cache_finished_req', 'cache_unfinished_req'):
            setattr(Segment, method, Segment.original)
        events = []
        modules = {'torch': SimpleNamespace(cuda=SimpleNamespace(
            nvtx=SimpleNamespace(range_push=lambda x: events.append(x), range_pop=lambda: events.append('pop')),
            profiler=SimpleNamespace(start=lambda: events.append('start'), stop=lambda: events.append('stop')))),
            'fixed_fa4_policy_hook': SimpleNamespace(make_hook=lambda config: lambda *args: None)}
        package = 'sglang.srt.speculative.dspark_components.'
        for path, name, cls in (
            (package + 'dspark_worker_v2', 'DSparkWorkerV2', Worker),
            (package + 'dspark_draft', 'DraftBlockProposer', Segment),
            (package + 'dspark_verify', 'TargetVerifyExecutor', Segment),
            (package + 'dspark_planner', 'DSparkVerifyPlanner', Segment),
            ('sglang.srt.mem_cache.unified_radix_cache', 'UnifiedRadixCache', Segment)):
            modules[path] = SimpleNamespace(**{name: cls})
        return modules, Worker, Segment, events

    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                tpot_trace_hook.make_hook({})

    def test_buffered_records_preserve_returns_and_no_profiler_in_host_mode(self):
        modules, Worker, Segment, events = self.modules()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {'SGLANG_DSPARK_TPOT_TRACE': '1'}):
            folder = Path(temp)
            config = {'diagnostic_folder': temp, 'nsys': False}
            tpot_trace_hook.make_hook(config)
            previous = Worker._forward_decode
            tpot_trace_hook.make_hook(config)
            self.assertIs(previous, Worker._forward_decode)
            req = SimpleNamespace(rid='rid', cache_protected_len=2047, origin_input_ids=[1]*2048)
            batch = SimpleNamespace(reqs=[req])
            worker = Worker()
            self.assertEqual(worker._forward_prefill(batch, None), ('prefill', batch))
            self.assertFalse(events)
            (folder / 'trace_start').touch()
            self.assertEqual(worker._forward_prefill(batch, None), ('prefill', batch))
            self.assertEqual(worker._forward_decode(batch, None), ('decode', batch))
            self.assertFalse((folder / 'worker_trace.jsonl').exists())
            (folder / 'trace_stop').touch()
            worker._forward_prefill(batch, None)
            records = [json.loads(x) for x in (folder/'worker_trace.jsonl').read_text().splitlines()]
            self.assertEqual([r['kind'] for r in records], ['prefill', 'decode'])
            self.assertEqual(records[-1]['step'], 1)
            self.assertEqual(records[-1]['protected_prefix'], 2047)
            self.assertNotIn('start', events)
            self.assertNotIn('stop', events)
            self.assertEqual(events.count('pop'), 2)

    def test_profiler_start_stop_once(self):
        modules, Worker, Segment, events = self.modules()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {'SGLANG_DSPARK_TPOT_TRACE': '1'}):
            folder = Path(temp)
            tpot_trace_hook.make_hook({'diagnostic_folder': temp, 'nsys': True})
            worker = Worker()
            batch = SimpleNamespace(reqs=[SimpleNamespace(rid='rid', cache_protected_len=0, origin_input_ids=[1])])
            (folder / 'trace_start').touch()
            worker._forward_prefill(batch, None)
            worker._forward_decode(batch, None)
            (folder / 'trace_stop').touch()
            worker._forward_prefill(batch, None)
            worker._forward_prefill(batch, None)
            self.assertEqual(events.count('start'), 1)
            self.assertEqual(events.count('stop'), 1)


if __name__ == '__main__':
    unittest.main()
