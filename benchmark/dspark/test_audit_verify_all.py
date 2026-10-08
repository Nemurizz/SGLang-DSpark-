import unittest
from audit_verify_all_ab import block_ratio_ci


class Tests(unittest.TestCase):
    def test_partial_window_has_no_confidence_interval(self):
        self.assertIsNone(block_ratio_ci([100] * 64, [99] * 63))

    def test_blocks_preserve_exact_constant_ratio(self):
        result = block_ratio_ci([100] * 64, [99.5] * 64, iterations=100)
        self.assertEqual(result['ratio'], .995)
        self.assertEqual(result['ci95_low'], .995)
        self.assertEqual(result['ci95_high'], .995)


if __name__ == '__main__':
    unittest.main()
