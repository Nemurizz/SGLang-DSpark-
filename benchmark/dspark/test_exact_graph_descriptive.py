import unittest
from exact_graph_descriptive_ab import PROTOCOL


class DescriptiveOnly(unittest.TestCase):
    def test_cannot_qualify_gain(self):
        self.assertTrue(PROTOCOL['exploratory_descriptive_only'])
        self.assertTrue(PROTOCOL['no_more_protocol_retries_after_this_run'])
        self.assertIn('DISABLED',PROTOCOL['performance_qualification'])

    def test_safety_and_load_unchanged(self):
        self.assertEqual(PROTOCOL['candidate_p95_tolerance'],0)
        self.assertEqual(PROTOCOL['request_retries'],0)
        self.assertFalse(PROTOCOL['profiler'])
        self.assertEqual(PROTOCOL['steady_requests_per_client'],32)
        self.assertEqual(PROTOCOL['cells'],['p2048_c4','p2048_c8','p2048_c16'])


if __name__=='__main__':unittest.main()
