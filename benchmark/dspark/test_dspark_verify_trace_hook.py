"""CPU metadata tests; model and torch are not needed."""
import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from dspark_verify_trace_hook import anchor_rows, make_hook, prompt_hash, request_rows


class TestTraceRows(unittest.TestCase):
    def test_hash_is_content_based(self):
        self.assertEqual(prompt_hash([1, 2]), prompt_hash([1, 2]))
        self.assertNotEqual(prompt_hash([1, 2]), prompt_hash([2, 1]))

    def test_live_lengths_not_padded_bucket(self):
        self.assertEqual(request_rows([2, 1, 3], 8), [0, 2, 3])
        with self.assertRaises(ValueError):
            request_rows([2, 0], 8)
        with self.assertRaises(ValueError):
            request_rows([8, 1], 8)

    def test_selects_committed_anchor_not_future_draft_row(self):
        kwargs = dict(lengths=[2, 1], pools=[4, 9], positions=[11, 12, 12],
                      ids=[7, 8, 8], pool_prompts={4: ("focus", 1), 9: ("focus", 2)},
                      focus_hash="focus", focus_position=12, anchor_token=8)
        self.assertEqual(anchor_rows(**kwargs), [(1, 2, 9)])
        kwargs["pool_prompts"] = {4: ("focus", 1), 9: ("different", 2)}
        self.assertEqual(anchor_rows(**kwargs), [])

    def test_rejects_mapping_and_position_mismatches(self):
        kwargs = dict(lengths=[2], pools=[4], positions=[12, 14], ids=[8, 9],
                      pool_prompts={4: ("focus", 1)}, focus_hash="focus",
                      focus_position=12, anchor_token=8)
        with self.assertRaises(ValueError):
            anchor_rows(**kwargs)
        kwargs["positions"] = [12, 13]
        kwargs["pools"] = [4, 9]
        with self.assertRaises(ValueError):
            anchor_rows(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch CPU required for hook-copy checks")
class TestReadOnlyHook(unittest.TestCase):
    def fake_graph_config(self, allowed=True):
        return SimpleNamespace(Backend=SimpleNamespace(DISABLED="off"),
                               Phase=SimpleNamespace(PREFILL="prefill", DECODE="decode"),
                               check_cuda_graph_backend=lambda *_: allowed)

    def test_snapshot_is_read_only_deduplicated_and_not_stale(self):
        import torch
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {
                "sglang.srt.model_executor.cuda_graph_config": self.fake_graph_config()}):
            hook = make_hook({"folder": folder, "prompt_sha256": prompt_hash([1, 2]),
                              "prompt_length": 2, "position": 2, "anchor_token": 3})
            mode = SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False)
            fb = SimpleNamespace(forward_mode=mode, req_pool_indices=torch.tensor([5]),
                                 extend_seq_lens_cpu=[2], extend_prefix_lens_cpu=[0])
            self.assertIsNone(hook(None, (torch.tensor([1, 2]), None, None, fb), None))
            mode.is_extend = lambda: False
            mode.is_target_verify = lambda: True
            mode.is_decode = lambda: False
            fb.extend_seq_lens_cpu = [1]
            fb.positions = torch.tensor([2])
            fb.seq_lens = torch.tensor([3])
            fb.seq_lens_cpu = torch.tensor([3])
            fb.out_cache_loc = torch.tensor([100])
            fb.spec_info = None
            hidden = torch.tensor([[1., 2.]])
            logits = torch.tensor([[0., 3., 2., 1.]])
            before_hidden, before_logits = hidden.clone(), logits.clone()
            args = (torch.tensor([3]), hidden, None, fb)
            output = SimpleNamespace(next_token_logits=logits)
            self.assertIsNone(hook(None, args, output))
            self.assertIsNone(hook(None, args, output))
            self.assertTrue(torch.equal(hidden, before_hidden))
            self.assertTrue(torch.equal(logits, before_logits))
            files = list(Path(folder).glob("*.json"))
            self.assertEqual(len(files), 1)
            metadata = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["raw_argmax"], 1)
            self.assertEqual(metadata["selected_row"], 0)
            tensors = torch.load(files[0].with_suffix(".pt"), weights_only=True)
            self.assertTrue(torch.equal(tensors["final_hidden"], hidden[0]))
            # Pool reuse must erase identity even for an incomplete prefill.
            mode.is_extend = lambda: True
            mode.is_target_verify = lambda: False
            fb.extend_seq_lens_cpu = [1]
            hook(None, (torch.tensor([99]), None, None, fb), None)
            mode.is_extend = lambda: False
            mode.is_target_verify = lambda: True
            hook(None, args, output)
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)

    def test_graph_enabled_is_refused(self):
        # Import before patch.dict: removing a newly imported torch from
        # sys.modules can trigger an unsafe second native-extension import.
        import torch
        with patch.dict("sys.modules", {"sglang.srt.model_executor.cuda_graph_config": self.fake_graph_config(False)}):
            with self.assertRaises(ValueError):
                make_hook({})


if __name__ == "__main__":
    unittest.main()
