"""CPU metadata tests; model and torch are not needed."""
import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from dspark_verify_trace_hook import (anchor_rows, make_hook, prompt_hash, request_rows,
                                    trace_stage, validate_past_locations, verify_query_lengths)


class TestTraceRows(unittest.TestCase):
    def test_stage_classification(self):
        attn = SimpleNamespace(layer_id=19, tp_q_head_num=40)
        self.assertEqual(trace_stage(attn), ("layer19.attention", "attention"))
        self.assertEqual(trace_stage(SimpleNamespace(self_attn=SimpleNamespace(attn=attn))),
                         ("layer19.post", "layer"))
        self.assertEqual(trace_stage(None), ("logits", "logits"))

    def test_past_kv_locations_checked_before_gpu_gather(self):
        validate_past_locations([4, 5], position=2, buffer_rows=10, current_locations=[6, 7])
        for locations in ([4], [4, 4], [-1, 4], [4, 10], [4, 6]):
            with self.assertRaises(ValueError):
                validate_past_locations(locations, position=2, buffer_rows=10, current_locations=[6, 7])

    def test_verify_metadata_does_not_assume_extend_lengths_exist(self):
        kwargs = dict(layout_lengths=[2, 1], declared_lengths=None,
                      fixed_tokens_per_request=8, pool_count=2, total_rows=4)
        self.assertEqual(verify_query_lengths(**kwargs), ([2, 1], "live_ragged_layout"))
        kwargs.update(layout_lengths=None, total_rows=16)
        self.assertEqual(verify_query_lengths(**kwargs)[0], [8, 8])
        kwargs["total_rows"] = 32
        with self.assertRaises(ValueError):
            verify_query_lengths(**kwargs)  # Never use a padded/stale bucket as live length.
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
            fb.extend_seq_lens_cpu = None
            fb.positions = torch.tensor([2])
            fb.seq_lens = torch.tensor([3])
            fb.seq_lens_cpu = torch.tensor([3])
            fb.out_cache_loc = torch.tensor([100])
            fb.spec_info = SimpleNamespace(draft_token_num=1, ragged_verify_layout=None)
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

    def test_upstream_stages_and_past_kv_are_read_only(self):
        import torch
        class FfiArrayLike:
            def __iter__(self):
                return iter([1, 2])
        keys = torch.arange(20, dtype=torch.float32).reshape(10, 1, 2)
        values = keys + 100
        req_map = torch.zeros((6, 3), dtype=torch.long)
        req_map[5] = torch.tensor([4, 5, 6])
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _: keys, get_value_buffer=lambda _: values),
            req_to_token_pool=SimpleNamespace(req_to_token=req_map),
            forward_metadata=SimpleNamespace(prefill_wrappers=[SimpleNamespace(_plan_info=FfiArrayLike())]))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {
                "sglang.srt.model_executor.cuda_graph_config": self.fake_graph_config(),
                "sglang.srt.model_executor.forward_context": SimpleNamespace(get_attn_backend=lambda: backend)}):
            config = {"folder": folder, "prompt_sha256": prompt_hash([1, 2]),
                      "prompt_length": 2, "position": 2, "anchor_token": 3,
                      "kv_layer": 0, "max_captures": 15}
            hook = make_hook(config)
            mode = SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False)
            fb = SimpleNamespace(forward_mode=mode, req_pool_indices=torch.tensor([5]),
                                 extend_seq_lens_cpu=[2], extend_prefix_lens_cpu=[0])
            hook(None, (torch.tensor([1, 2]), None, None, fb), None)
            mode.is_extend = lambda: False
            mode.is_target_verify = lambda: True
            mode.is_decode = lambda: False
            fb.extend_seq_lens_cpu = None
            fb.input_ids = torch.tensor([3])
            fb.positions = torch.tensor([2])
            fb.seq_lens = fb.seq_lens_cpu = torch.tensor([3])
            fb.out_cache_loc = torch.tensor([6])
            fb.spec_info = SimpleNamespace(draft_token_num=1, ragged_verify_layout=None)
            attn = SimpleNamespace(layer_id=0, tp_q_head_num=1)
            layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
            q, k, v, out = [torch.tensor([[float(i), 2.]]) for i in range(4)]
            originals = [value.clone() for value in (q, k, v, out, keys, values, req_map)]
            for _ in range(2):
                self.assertIsNone(hook(attn, (q, k, v, fb), out))
                self.assertIsNone(hook(layer, (fb.positions, q, fb, None), (out, v)))
            files = list(Path(folder).glob("*.json"))
            self.assertEqual(len(files), 2)  # Deduplicate per epoch AND stage, not just epoch.
            records = {json.loads(path.read_text())["stage"]: path for path in files}
            tensors = torch.load(records["layer0.attention"].with_suffix(".pt"), weights_only=True)
            self.assertTrue(torch.equal(tensors["past_key"], keys[[4, 5]]))
            self.assertTrue(torch.equal(tensors["written_key"], keys[6]))
            self.assertTrue(torch.equal(tensors["query"], q[0]))
            self.assertEqual(json.loads(records["layer0.attention"].read_text())["wrapper_plan_info"], [[1, 2]])
            for value, original in zip((q, k, v, out, keys, values, req_map), originals):
                self.assertTrue(torch.equal(value, original))
            with self.assertRaises(ValueError):
                make_hook({**config, "max_kv_prefix": 1})


if __name__ == "__main__":
    unittest.main()
