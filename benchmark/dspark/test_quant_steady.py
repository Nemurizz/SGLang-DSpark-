import unittest
from quant_steady_ab import noise, admitted


class SteadyTests(unittest.TestCase):
    def test_control_gate_checks_e2e_and_tpot(self):
        a = {key: {'mean': 100, 'p95': 101} for key in ('tpot_ms', 'e2e_ms')}
        b = {key: {'mean': 100.3, 'p95': 101.3} for key in ('tpot_ms', 'e2e_ms')}
        self.assertTrue(noise(a, b)['admitted'])
        b['e2e_ms']['mean'] = 101
        self.assertFalse(noise(a, b)['admitted'])

    def test_warmup_minimum_and_spread(self):
        group = {'mean': 100, 'p50': 100, 'p95': 101}
        self.assertFalse(admitted([group, group]))
        self.assertTrue(admitted([group]*3))
        self.assertFalse(admitted([group, group, {**group, 'mean': 101}]))


if __name__ == '__main__':
    unittest.main()
