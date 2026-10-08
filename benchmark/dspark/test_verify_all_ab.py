import copy
import unittest

import dspark_prefix_ab as ab
from verify_all_ab import PROTOCOL, comparison, migrated_template, noise_gate


def summary(mean=100, p95=101):
    return {'request_count': 64, 'cell': 'p2048_c1',
        **{key: {'mean': mean, 'p95': p95} for key in ('ttft_ms', 'tpot_ms', 'e2e_ms')}}


class Tests(unittest.TestCase):
    def test_migration_is_explicit_and_preserves_original(self):
        source = {'qualified_manifest': {'gpu_identity': 'SM120, oldUUID, 595.71.05'}}
        old = copy.deepcopy(source)
        migrated = migrated_template(source, 'SM120, newUUID, 595.71.05')
        self.assertEqual(source, old)
        self.assertNotEqual(migrated, source)

    def test_different_driver_or_gpu_rejected(self):
        source = {'qualified_manifest': {'gpu_identity': 'SM120, oldUUID, 595.71.05'}}
        for identity in ('SM120, UUID, otherDriver', 'B200, UUID, 595.71.05'):
            with self.subTest(identity=identity), self.assertRaises(ab.SafetyFailure):
                migrated_template(source, identity)

    def test_no_relaxed_p95_or_retries(self):
        self.assertEqual(PROTOCOL['candidate_p95_tolerance_ttft_tpot_e2e'], 0)
        self.assertEqual(PROTOCOL['request_retries'], 0)
        self.assertFalse(PROTOCOL['profiler'])

    def test_noise_guard(self):
        self.assertTrue(noise_gate(summary(), summary(100.1, 101.1))['admitted'])
        self.assertFalse(noise_gate(summary(), summary(101, 103))['admitted'])

    def test_one_percent_threshold(self):
        self.assertTrue(comparison(summary(), summary(99, 100))['e2e_gain_at_least_one_percent'])
        self.assertFalse(comparison(summary(), summary(99.5, 100))['e2e_gain_at_least_one_percent'])

    def test_p95_regression_stops_even_with_mean_gain(self):
        for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
            candidate = summary(98, 100)
            candidate[metric]['p95'] = 101.01
            with self.subTest(metric=metric), self.assertRaises(ab.SafetyFailure):
                comparison(summary(), candidate)


if __name__ == '__main__':
    unittest.main()
