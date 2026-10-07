import unittest

from decompose_baseline_tail import correlation, union_ns


class TailTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(union_ns([]), 0)

    def test_overlap_and_nested(self):
        self.assertEqual(union_ns([(0, 2), (1, 3), (1, 2), (7, 9)]), 5)

    def test_touching_and_unsorted(self):
        self.assertEqual(union_ns([(2, 4), (0, 2)]), 4)

    def test_negative_correlation(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1)

    def test_constant_has_no_correlation(self):
        self.assertIsNone(correlation([1, 1], [2, 3]))


if __name__ == '__main__':
    unittest.main()
