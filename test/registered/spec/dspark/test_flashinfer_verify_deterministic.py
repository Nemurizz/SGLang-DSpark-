"""Metadata regression: eager target verify forwards deterministic split size.

No model or CUDA tensors are created. Native backend imports need GPU wheels.
CUDA graph planning is intentionally outside this test's scope.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TestFlashInferVerifyDeterministic(unittest.TestCase):
    def test_eager_verify_passes_configured_split_size(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        for split_size in (None, 4096, 2048):
            with self.subTest(split_size=split_size):
                updater = SimpleNamespace(update=Mock())
                backend = SimpleNamespace(use_sliding_window_kv_pool=False,
                                          indices_updater_prefill=updater,
                                          prefill_wrappers_verify=[object()],
                                          prefill_split_tile_size=split_size)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False,
                                                 is_target_verify=lambda: True),
                    req_pool_indices=object(), seq_lens=object(), seq_lens_cpu=object(),
                    seq_lens_sum=123, encoder_lens=None, spec_info=object())
                FlashInferAttnBackend.init_forward_metadata(backend, batch)
                call = updater.update.call_args
                self.assertIn("fixed_split_size", call.kwargs)
                self.assertEqual(call.kwargs["fixed_split_size"], split_size)
                self.assertIs(call.kwargs["spec_info"], batch.spec_info)
                self.assertFalse(call.kwargs["use_ragged"])
                self.assertEqual(backend.forward_metadata.prefill_wrappers,
                                 backend.prefill_wrappers_verify)


if __name__ == "__main__":
    unittest.main()
