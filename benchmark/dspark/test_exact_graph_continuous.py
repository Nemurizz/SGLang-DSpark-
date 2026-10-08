import ast
from pathlib import Path
import unittest
from exact_graph_continuous_ab import PROTOCOL,oracle_cell


class ContinuousProtocol(unittest.TestCase):
    def test_frozen_mapping_no_new_oracle(self):
        self.assertEqual(oracle_cell('p2048_c4'),'p2048_c4')
        self.assertEqual(oracle_cell('p2048_c8'),'p2048_c8')
        self.assertEqual(oracle_cell('p2048_c16'),'p2048_c8')
        self.assertEqual([s%8 for s in range(16)],list(range(8))*2)

    def test_fixed_load_sample_count_and_gates(self):
        self.assertEqual(PROTOCOL['steady_requests_per_client'],32)
        self.assertEqual(PROTOCOL['steady_request_count_by_concurrency'],{'4':128,'8':256,'16':512})
        self.assertEqual(PROTOCOL['warmup_fixed_groups'],3)
        self.assertEqual(PROTOCOL['warmup_mean_spread_limit'],.005)
        self.assertEqual(PROTOCOL['candidate_p95_tolerance'],0)
        self.assertEqual(PROTOCOL['request_retries'],0)
        self.assertFalse(PROTOCOL['profiler'])

    def test_replenishment_is_inside_each_client_not_barrier_per_iteration(self):
        tree=ast.parse(Path(__file__).with_name('exact_graph_continuous_ab.py').read_text())
        client=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='client')
        loop=next(n for n in ast.walk(client) if isinstance(n,ast.For))
        self.assertTrue(any(isinstance(n,ast.Await) for n in ast.walk(loop)))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='wave' for n in ast.walk(client)))


if __name__=='__main__':unittest.main()
