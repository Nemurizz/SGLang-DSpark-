import unittest
from exact_graph_concurrent_ab import PROTOCOL, batch_labels, ratios


class ConcurrentProtocol(unittest.TestCase):
    def test_complete_distinct_wave(self):
        for c in (4,8):
            self.assertEqual(batch_labels('p2048_c'+str(c),2,list(range(c)))['variants'],list(range(c)))
            with self.assertRaises(ValueError):batch_labels('p2048_c'+str(c),2,[0]*c)
            with self.assertRaises(ValueError):batch_labels('p2048_c'+str(c),2,list(range(c-1)))

    def test_no_numerical_or_geometry_expansion(self):
        self.assertEqual(PROTOCOL['cells'],['p2048_c4','p2048_c8'])
        self.assertEqual(PROTOCOL['request_retries'],0)
        self.assertFalse(PROTOCOL['profiler'])
        self.assertFalse(PROTOCOL['diagnostic_hooks'])
        self.assertEqual(PROTOCOL['candidate_p95_tolerance'],0)
        self.assertEqual(PROTOCOL['steady_waves_per_window_per_cell'],16)
        self.assertEqual(PROTOCOL['order'],['baseline','target_graph','target_graph','baseline'])

    def test_no_mean_gain_masks_tail_regression(self):
        a={m:{'mean':100,'p95':100} for m in ('ttft_ms','tpot_ms','e2e_ms')}
        b={m:{'mean':98,'p95':101} for m in ('ttft_ms','tpot_ms','e2e_ms')}
        r=ratios(a,b)
        self.assertLessEqual(r['e2e_ms']['mean'],.99)
        self.assertTrue(all(v['p95']>1 for v in r.values()))


if __name__=='__main__':unittest.main()
