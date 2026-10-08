import tempfile
import unittest
from types import SimpleNamespace as NS

from verify_all_policy_hook import POLICY, apply_to_planner, bind_existing, install, validate_options


def options():
    return NS(tp_size=1, pp_size=1, dp_size=1, enable_dp_attention=False,
        disable_cuda_graph=True, enable_deterministic_inference=True, page_size=1,
        speculative_algorithm='DSPARK', speculative_num_draft_tokens=8,
        speculative_dspark_block_size=7, speculative_dspark_sps_table_path=None)


def env():
    return {'SGLANG_DSPARK_FIXED_VERIFY_ALL_EXPERIMENT': '1',
        'SGLANG_RAGGED_VERIFY_MODE': 'compact', 'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH': '0',
        'SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH': '0', 'SGLANG_DSPARK_COMPACT_SCAN_CUDA': '0'}


def planner_class():
    class Planner:
        def __init__(self, **kwargs):
            self.server_args = kwargs.get('server_args', options())
            self._is_verify_all = self.is_compact_mode = self.carries_confidence = True
            self._dp_tier_gather_enabled = False
            self.gamma, self.verify_num_draft_tokens = 7, 8
            self._schedule_cfg = NS(max_verify_len=0)
            self.model_runner = NS(decode_cuda_graph_runner=None)
            self._uniform_layout_cache = {}
            self._budget_planner = object()
            self._dynamic_graph_tier = True

        @property
        def schedules_verify_budget(self):
            return self._budget_planner is not None

        def confidence_budget_prepare(self):
            return 'callback' if self.schedules_verify_budget else None
    return Planner


class Tests(unittest.TestCase):
    def test_default_off(self):
        with self.assertRaises(ValueError):
            install(planner_class(), {}, {})

    def test_options_fail_closed(self):
        for key, value in [('tp_size', 2), ('enable_dp_attention', True),
            ('disable_cuda_graph', False), ('page_size', 16),
            ('speculative_dspark_sps_table_path', '/old/table.json')]:
            a = options()
            setattr(a, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_options(a, env())

    def test_dynamic_and_collection_rejected(self):
        for key in ('SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH', 'SGLANG_DSPARK_STS_COLLECT_PATH'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_options(options(), {**env(), key: '1'})

    def test_adaptive_full_budget_is_not_uniform(self):
        p = planner_class()()
        p._is_verify_all = False
        with self.assertRaises(ValueError):
            apply_to_planner(p)

    def test_live_layout_or_graph_rejected(self):
        for key, value in [('_uniform_layout_cache', {1: object()}),
                           ('model_runner', NS(decode_cuda_graph_runner=object()))]:
            p = planner_class()()
            setattr(p, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_to_planner(p)

    def test_idempotence_preserves_head_and_records_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            cls = planner_class()
            config = {'profile_id': 'frozen', 'policy_folder': folder, 'verify_policy': POLICY}
            marker = install(cls, config, env())
            self.assertIs(install(cls, config, env()), marker)
            p = cls(server_args=options())
            self.assertIsNone(p.confidence_budget_prepare())
            self.assertTrue(p.carries_confidence)
            self.assertEqual(marker['planner_instances'], 1)
            with self.assertRaises(ValueError):
                install(cls, {**config, 'profile_id': 'different'}, env())

    def test_late_factory_binds_existing_planner_once(self):
        with tempfile.TemporaryDirectory() as folder:
            cls = planner_class()
            p = cls(server_args=options())
            config = {'profile_id': 'frozen', 'policy_folder': folder}
            marker = install(cls, config, env())
            bind_existing(cls, marker, config, env(), [object(), p])
            bind_existing(cls, marker, config, env(), [p])
            self.assertEqual(marker['planner_instances'], 1)
            self.assertIsNone(p.confidence_budget_prepare())

    def test_late_factory_missing_or_multiple_planners_fail_closed(self):
        for number in (0, 2):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as folder:
                cls = planner_class()
                planners = [cls(server_args=options()) for _ in range(number)]
                config = {'profile_id': 'frozen', 'policy_folder': folder}
                marker = install(cls, config, env())
                with self.assertRaises(ValueError):
                    bind_existing(cls, marker, config, env(), planners)


if __name__ == '__main__':
    unittest.main()
