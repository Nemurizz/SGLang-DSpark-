import unittest

from steady_tpot import PROTOCOL, admitted, control_check


class StabilityTests(unittest.TestCase):
    def group(self, value=19, p95=19.1):
        return {'mean': value, 'p50': 19, 'p95': p95}

    def test_requires_three_groups(self):
        self.assertFalse(admitted([self.group(), self.group()]))

    def test_drift_rejected(self):
        self.assertFalse(admitted([self.group(20), self.group(), self.group()]))

    def test_tail_rejected(self):
        self.assertFalse(admitted([self.group(), self.group(), self.group(p95=20)]))

    def test_stationary_admitted(self):
        self.assertTrue(admitted([self.group(19.01), self.group(19), self.group(19.02)]))

    def test_noise_admission_is_symmetric(self):
        a = {'tpot_ms': {'mean': 19, 'p95': 19.1}}
        b = {'tpot_ms': {'mean': 19.01, 'p95': 19.15}}
        self.assertEqual(control_check(a, b), control_check(b, a))
        self.assertTrue(control_check(a, b)['admitted'])

    def test_drift_blocks_candidate(self):
        a = {'tpot_ms': {'mean': 19, 'p95': 19.1}}
        b = {'tpot_ms': {'mean': 20, 'p95': 20.1}}
        self.assertFalse(control_check(a, b)['admitted'])

    def test_candidate_tolerance_is_zero(self):
        self.assertEqual(PROTOCOL['candidate_p95_tolerance'], 0)
        self.assertEqual(PROTOCOL['candidate_order'], ['full', 'paired', 'paired', 'full'])


if __name__ == '__main__':
    unittest.main()
