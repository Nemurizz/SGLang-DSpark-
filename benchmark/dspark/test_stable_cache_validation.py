import unittest
from stable_cache_validation import balanced_order, bootstrap_ratio, pressure_prompts


class ProtocolTests(unittest.TestCase):
    def test_ab_ba(self):
        self.assertEqual(balanced_order(2), [('full', 'paired'), ('paired', 'full')])

    def test_reject_unbalanced(self):
        for pairs in (0, 1, 3):
            with self.assertRaises(ValueError):
                balanced_order(pairs)

    def test_disjoint_pressure(self):
        base = list(range(8192))
        prompts = pressure_prompts({'p8192_c8': {'prompts': [base]}})
        self.assertEqual(len({row[0] for row in prompts}), 6)
        self.assertTrue(all(len(row) == 8192 and row[1:] == base[1:] for row in prompts))

    def test_cluster_gain(self):
        result = bootstrap_ratio([100.] * 8, [50.] * 8, 42, 100)
        self.assertEqual(result['ci95_high'], .5)
        self.assertEqual(result['ratio'], .5)

    def test_insufficient_clusters(self):
        self.assertIsNone(bootstrap_ratio([100.], [50.], 42))


if __name__ == '__main__':
    unittest.main()
