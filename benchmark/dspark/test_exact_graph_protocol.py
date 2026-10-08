"""CPU fail-closed tests: no-effect, actual layout, noise and zero-p95 gates."""
import json
from pathlib import Path
import tempfile
import unittest

import dspark_prefix_ab as ab
from exact_target_graph_ab import PROTOCOL, noise_gate, validate_graph_attestation


def summary(value=1):
    return {'request_count': 64, 'cell': 'p2048_c1',
        **{m: {'mean': value, 'p95': value} for m in ('ttft_ms', 'tpot_ms', 'e2e_ms')}}


class Protocol(unittest.TestCase):
    def fixture(self, folder):
        policy=folder/'graph_policy'
        policy.mkdir()
        records={
            'graph_policy.json': {'enabled':True,'diagnostic':False,'process_local':True,
                'target_capture_tokens':[8],'draft_graph':False,'prefill_graph':False,
                'current_device_host_mirror':True},
            'capture_actual.json': {'bs':1,'verify_lens_cpu':[8],'tokens':[8],'buffer_max_bs':8},
            'draft_actual.json': {'decode_graph_allowed':False},
            'graph_calls.json': {'actual_execute':1,'actual_execute_by_bs':{'1':1}},
            'graph_warmup.json': {'actual_execute':1},
            'graph_final.json': {'actual_execute':100,'actual_execute_by_bs':{'1':100}}}
        for name,record in records.items(): (policy/name).write_text(json.dumps(record))
        return records

    def test_actual_capture_and_final(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            validate_graph_attestation(folder,False)
            with self.assertRaises(ab.SafetyFailure): validate_graph_attestation(folder,True)
            self.fixture(folder)
            validate_graph_attestation(folder,True)
            validate_graph_attestation(folder,True,final=True)
            with self.assertRaises(ab.SafetyFailure): validate_graph_attestation(folder,False)

    def test_no_effect_wrong_layout_or_draft_never_admitted(self):
        changes=[('graph_calls.json','actual_execute',0),('graph_calls.json','actual_execute_by_bs',{'4':1}),
            ('capture_actual.json','bs',8),('capture_actual.json','verify_lens_cpu',[1]*8),
            ('capture_actual.json','buffer_max_bs',1),('draft_actual.json','decode_graph_allowed',True),
            ('graph_policy.json','diagnostic',True)]
        for filename,key,value in changes:
            with self.subTest(filename=filename,key=key),tempfile.TemporaryDirectory() as name:
                folder=Path(name)
                records=self.fixture(folder)
                records[filename][key]=value
                (folder/'graph_policy'/filename).write_text(json.dumps(records[filename]))
                with self.assertRaises(ab.SafetyFailure): validate_graph_attestation(folder,True)

    def test_warmup_only_executions_do_not_qualify(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            records=self.fixture(folder)
            records['graph_final.json']['actual_execute']=64
            (folder/'graph_policy'/'graph_final.json').write_text(json.dumps(records['graph_final.json']))
            with self.assertRaises(ab.SafetyFailure): validate_graph_attestation(folder,True,final=True)

    def test_noise_and_p95_thresholds_unchanged(self):
        self.assertTrue(noise_gate(summary(),summary(1.004))['admitted'])
        self.assertFalse(noise_gate(summary(),summary(1.006))['admitted'])
        item=summary(.98)
        item['ttft_ms']['p95']=1.000001
        with self.assertRaises(ab.SafetyFailure): ab.check_p95(summary(),item,64,0)
        self.assertEqual(ab.check_p95(summary(),summary(.98),64,0),'passed')

    def test_prespecified_no_profiler_no_retry_and_both_orders(self):
        self.assertEqual(PROTOCOL['candidate_order'],['baseline','target_graph','target_graph','baseline'])
        self.assertEqual(PROTOCOL['minimum_e2e_gain_each_order'],.01)
        self.assertEqual(PROTOCOL['candidate_p95_tolerance_ttft_tpot_e2e'],0)
        self.assertEqual(PROTOCOL['steady_requests_per_window'],64)
        self.assertEqual(PROTOCOL['correctness_cells'],['p2048_c1','p2048_c4','p2048_c8'])
        self.assertFalse(PROTOCOL['profiler'])
        self.assertFalse(PROTOCOL['diagnostic_hooks'])
        self.assertEqual(PROTOCOL['request_retries'],0)


if __name__=='__main__': unittest.main()
