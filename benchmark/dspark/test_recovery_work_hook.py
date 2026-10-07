import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from recovery_work_hook import validate_work


class WorkTests(unittest.TestCase):
    def row(self):
        return dict(prompt_tokens=8192, actual_prefix_tokens=8172,
            actual_prefill_tokens=20, draft_injected_tokens=20, target_hit_tokens=8191,
            paired_hit_tokens=8172, planned_recovery_tokens=19,
            actual_recovered_tokens=19, plan_mode='partial')

    def test_partial(self):
        validate_work(self.row())

    def test_not_plan_only(self):
        row = self.row()
        row['actual_prefix_tokens'] = 0
        with self.assertRaises(ValueError):
            validate_work(row)

    def test_injection_count(self):
        row = self.row()
        row['draft_injected_tokens'] = 0
        with self.assertRaises(ValueError):
            validate_work(row)

    def test_real_wrapper_contract_and_success_record(self):
        from recovery_work_hook import make_hook

        class Injector:
            def inject_target_hidden(self, **kwargs):
                self.written = kwargs['cache_loc'].numel()

        class Worker:
            def _forward_prefill(self, batch, on_publish):
                self.injector.inject_target_hidden(target_hidden=SimpleNamespace(shape=(20, 1)),
                    cache_loc=SimpleNamespace(numel=lambda: 20), positions=SimpleNamespace(numel=lambda: 20))
                batch.reqs[0].dspark_draft_kv_ready = True
                return 'original-result'

        modules = {
            'fixed_fa4_policy_hook': SimpleNamespace(make_hook=lambda config: 'attestation'),
            'sglang.srt.speculative.dspark_components.dspark_worker_v2': SimpleNamespace(DSparkWorkerV2=Worker),
            'sglang.srt.speculative.dspark_components.dspark_kv_inject': SimpleNamespace(TargetHiddenKvInjector=Injector)}
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', modules), patch.dict(
                'os.environ', SGLANG_DSPARK_RECOVERY_WORK_OBSERVER='1'):
            config = dict(profile_id='test', work_folder=folder)
            self.assertEqual(make_hook(config), 'attestation')
            self.assertEqual(make_hook(config), 'attestation')
            req = SimpleNamespace(rid='request', origin_input_ids=[1] * 8192,
                dspark_target_prefix_len=8191, dspark_paired_prefix_len=8172,
                dspark_recovery_tokens=19, dspark_recovery_mode='partial', dspark_draft_kv_ready=False)
            batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False),
                reqs=[req], prefix_lens=[8172], extend_lens=[20])
            worker = Worker()
            worker.injector = Injector()
            self.assertEqual(worker._forward_prefill(batch, None), 'original-result')
            rows = [json.loads(line) for line in (Path(folder) / 'worker_work.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['actual_recovered_tokens'], 19)
            self.assertEqual(rows[0]['draft_injected_tokens'], 20)
            self.assertTrue(rows[0]['draft_ready_after_injection'])


if __name__ == '__main__':
    unittest.main()
