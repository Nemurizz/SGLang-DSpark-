import ast
from pathlib import Path
import unittest
import original_norm_graph_ab as ab


class Protocol(unittest.TestCase):
    def test_order_and_threshold(self):
        p=ab.PROTOCOL
        self.assertEqual(p['candidate_order'], ['baseline','graph','graph','baseline'])
        self.assertEqual(p['minimum_e2e_gain_each_order'], .01)
        self.assertEqual(p['candidate_p95_tolerance_ttft_tpot_e2e'], 0.0)
        self.assertEqual(p['steady_requests_per_window'],64)

    def test_no_retry_instrumentation(self):
        p=ab.PROTOCOL
        self.assertFalse(p['profiler'])
        self.assertFalse(p['diagnostic_hooks'])
        self.assertEqual(p['request_retries'],0)

    def test_guard_scope(self):
        source=Path(__file__).with_name('residual_norm_graph.py').read_text()
        self.assertIn('torch.cuda.is_current_stream_capturing()',source)
        self.assertIn('weight._version',source)
        self.assertIn('self.sealed',source)
        self.assertNotIn('pool=',source)
        self.assertNotIn('matmul',source)

    def test_lifetimes_and_stream(self):
        source=Path(__file__).with_name('residual_norm_graph.py').read_text()
        self.assertIn('entry = (graph, static_x, static_r, output, side)',source)
        self.assertIn('cuda_stream',source)
        self.assertIn('static_x.copy_(x)',source)
        self.assertIn('static_r.copy_(residual)',source)

    def test_steady_sealed(self):
        source=Path(ab.__file__).read_text()
        self.assertLess(source.index("'seal_catalog'"),source.index('for window in range(windows)'))
        self.assertIn("final['captures'] != sealed['captures']",source)
        self.assertIn('ab.check_p95(control, candidate, 64, 0.0)',source)

    def test_python_sources(self):
        for path in Path(__file__).parent.glob('*norm_graph*.py'):
            ast.parse(path.read_text())


if __name__ == '__main__':
    unittest.main()
