import unittest

from fixed_fa4_policy_hook import locked_options


class TestFixedPolicy(unittest.TestCase):
    def options(self):
        return dict(head_dim=128, head_dim_v=128, num_head_kv=8, qhead_per_kvhead=5,
                    has_bias=False, is_local=False, is_causal=True, pack_gqa=True, tile_mn=None)

    def test_policy_is_explicit_and_preserves_all_other_metadata(self):
        options = self.options()
        options.update(total_q_rows=320, seqlen_q=8, seqlen_k=8192, paged_kv=True, num_batch=4)
        changed = locked_options(options)
        self.assertIsNone(options['tile_mn'])
        self.assertEqual(changed, {**options, 'tile_mn': (64,64)})

    def test_unqualified_configs_fail_closed(self):
        for name, value in (('head_dim',64), ('head_dim_v',64), ('num_head_kv',4),
                            ('qhead_per_kvhead',8), ('has_bias',True), ('is_local',True),
                            ('is_causal',False), ('pack_gqa',False), ('tile_mn',(64,128))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                locked_options({**self.options(),name:value})


if __name__ == '__main__':
    unittest.main()
