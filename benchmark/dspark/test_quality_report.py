import unittest
from quality_report import permitted, paired_interval


class ReportTests(unittest.TestCase):
    def test_human_one_percentage_point_budget(self):
        self.assertTrue(permitted(120, 119, 128))
        self.assertFalse(permitted(120, 118, 128))
        self.assertFalse(permitted(60, 59, 64))
        self.assertTrue(permitted(100, 99, 100))
        self.assertTrue(permitted(90, 92, 100))

    def test_paired_identical_outcomes_interval(self):
        self.assertEqual(paired_interval([0]*128, draws=1000), [0, 0])


if __name__ == '__main__':
    unittest.main()
