"""CPU tests of the pinned FA2 B=1 partition derivation, no model loading."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[2] / "python/sglang/srt/layers/attention/dspark_reference_attention.py"
if not source.exists():
    source = Path(__file__).resolve().parents[1] / "theory_review/dspark_reference_attention.py"
spec = importlib.util.spec_from_file_location("dspark_reference_attention", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestReferencePartition(unittest.TestCase):
    def test_matches_installed_single_request_binary_search(self):
        capacity = 47  # Recorded RTX PRO 6000 SM=188, 8 KV heads.
        for length in (8, 32, 127, 128, 129, 512, 2061, 6016, 6017, 8200, 8320, 32768):
            low, high = 128, length
            while low < high:
                mid = (low + high) // 2
                if (length + mid - 1) // mid > capacity:
                    low = mid + 1
                else:
                    high = mid
            for query in range(1, min(length, 8) + 1):
                self.assertEqual(module.canonical_groups([query], [length], 188),
                                 [(0, 1, low, 16 if query <= 3 else 64)])

    def test_grouping_is_contiguous_and_preserves_each_request_rule(self):
        self.assertEqual(module.canonical_groups([8] * 8, [2061] * 8, 188), [(0, 8, 128, 64)])
        lengths = [8192, 8193, 8226, 8192]
        groups = module.canonical_groups([8, 4, 3, 1], lengths, 188)
        self.assertEqual(groups, [(0, 2, 175, 64), (2, 3, 176, 16), (3, 4, 175, 16)])
        self.assertEqual(module.canonical_groups([8, 1, 8], [2061] * 3, 188),
                         [(0, 1, 128, 64), (1, 2, 128, 16), (2, 3, 128, 64)])
        # Lengths are live, not a padded bucket; zero is not accepted.
        for queries, kv in (([], []), ([0], [100]), ([9], [100]), ([2], [1]), ([1, 2], [100])):
            with self.assertRaises(ValueError):
                module.canonical_groups(queries, kv, 188)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch CPU required for wrapper slicing test")
    def test_wrapper_slicing_uses_live_indptr_and_output_order(self):
        import torch
        plans = []
        class FakeWrapper:
            def __init__(self, *_args, **_kwargs):
                pass
            def plan(self, qo, kv, indices, last, *_args, **kwargs):
                plans.append((qo.clone(), kv.clone(), indices.clone(), kwargs["fixed_split_size"]))
                cta = 16 if int(qo[-1]) * 5 // (len(qo) - 1) <= 16 else 64
                self._plan_info = [1, int(qo[-1]), 0, cta] + [0] * 11
            def forward(self, q, *_args, **_kwargs):
                return q.clone()
        original = SimpleNamespace(is_cuda_graph_enabled=False, _custom_mask_buf=None,
            _qo_indptr_buf=torch.tensor([0, 8, 12, 15]),
            _paged_kv_indptr_buf=torch.tensor([0, 8192, 16385, 24611]),
            _paged_kv_indices_buf=torch.arange(24611, dtype=torch.int32),
            _paged_kv_last_page_len_buf=torch.ones(3, dtype=torch.int32))
        fb = SimpleNamespace(input_ids=torch.zeros(15), req_pool_indices=torch.arange(3))
        policy = module.SerialReferencePlans.__new__(module.SerialReferencePlans)
        policy.sm_count, policy.wrappers = 188, []
        policy.backend = SimpleNamespace(workspace_buffer=None,
            indices_updater_prefill=SimpleNamespace(q_data_type=torch.bfloat16, data_type=torch.bfloat16))
        with patch.dict("sys.modules", {"flashinfer": SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=FakeWrapper)}):
            groups = policy.plan(original, fb)
            self.assertEqual([(start, end) for start, end, _ in groups], [(0, 12), (12, 15)])
            self.assertEqual([plan[3] for plan in plans], [175, 176])
            self.assertEqual(plans[0][0].tolist(), [0, 8, 12])
            self.assertEqual(plans[1][1].tolist(), [0, 8226])
            self.assertEqual(plans[1][2][0].item(), 16385)
            q = torch.arange(30).reshape(15, 2)
            self.assertTrue(torch.equal(module.forward_groups(groups, q, None), q))
            fb.input_ids = torch.zeros(16)  # Padded/stale bucket is NOT a live query length.
            with self.assertRaises(ValueError):
                policy.plan(original, fb)


if __name__ == "__main__":
    unittest.main()
