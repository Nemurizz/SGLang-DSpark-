"""CPU-only tests for fail-closed admission and PID-qualified trace attribution."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analyze_norm_trace import analyze
from residual_norm_ab import PROTOCOL, noise_gate, validate_norm_attestation
import dspark_prefix_ab as ab


def summary(value=1):
    return {'request_count': 64, 'cell': 'p2048_c1',
            **{m: {'mean': value, 'p95': value} for m in ('ttft_ms', 'tpot_ms', 'e2e_ms')}}


class ProtocolTests(unittest.TestCase):
    def test_noise_gate_covers_e2e_and_tpot(self):
        self.assertTrue(noise_gate(summary(), summary(1.004))['admitted'])
        self.assertFalse(noise_gate(summary(), summary(1.006))['admitted'])
        item = summary()
        item['e2e_ms']['p95'] = 1.011
        self.assertFalse(noise_gate(summary(), item)['admitted'])

    def test_p95_has_zero_candidate_tolerance(self):
        item = summary(.98)
        item['ttft_ms']['p95'] = 1.000001
        with self.assertRaises(ab.SafetyFailure):
            ab.check_p95(summary(), item, 64, PROTOCOL['candidate_p95_tolerance_ttft_tpot_e2e'])
        self.assertEqual(ab.check_p95(summary(), summary(.98), 64, 0), 'passed')

    def test_runtime_attestation_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            validate_norm_attestation(folder, False)
            with self.assertRaises(ab.SafetyFailure):
                validate_norm_attestation(folder, True)
            policy = folder / 'norm_policy'
            policy.mkdir()
            (policy / 'norm_policy.json').write_text(json.dumps(
                {'enabled': True, 'diagnostic': False, 'process_local': True}))
            (policy / 'norm_calls.json').write_text(json.dumps({'eligible_calls': 1}))
            validate_norm_attestation(folder, True)
            with self.assertRaises(ab.SafetyFailure):
                validate_norm_attestation(folder, False)

    def fixture(self, folder, steps=40):
        (folder / 'requests.jsonl').write_text(json.dumps(
            {'phase': 'diagnostic_profiled', 'meta_info': {'id': 'rid'}}) + '\n')
        conn = sqlite3.connect(folder / 'trace.sqlite')
        conn.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
        ''')
        conn.execute('INSERT INTO StringIds VALUES(1,"norm")')
        pid, tid = 7 << 24, (7 << 24) + 1
        for step in range(steps):
            start = step * 10000
            conn.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)',
                (start, start + 9000, tid, 'DSpark._forward_decode ' + json.dumps({'rid': 'rid', 'step': step})))
            conn.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)',
                (start + 1, start + 8000, tid, 'DSparkResidualNorm ' + json.dumps({'candidate': False})))
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?)', (start + 2, tid, step))
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,1)',
                (start + 10, start + 1010, pid, step))
            # Same correlation ID in a different process must NOT be attributed.
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,1)',
                (start + 10, start + 2010, 8 << 24, step))
        conn.commit()
        conn.close()

    def test_trace_requires_pid_qualified_launch(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            result = analyze(folder)
            self.assertEqual(result['actual_decode_steps'], 40)
            self.assertEqual(result['norm_kernels_per_decode'], 1)
            self.assertEqual(result['norm_gpu_us_per_decode'], 1)

    def test_trace_missing_steps_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder, steps=39)
            with self.assertRaises(ValueError):
                analyze(folder)


if __name__ == '__main__':
    unittest.main()
