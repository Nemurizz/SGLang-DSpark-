import unittest
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import profile_command, validate_profile, validate_row
from stable_profile_hook import validate_runtime


class TestStableGate(unittest.TestCase):
    def command(self):
        cmd = ["python", "-m", "sglang.launch_server", "--attention-backend", "flashinfer",
               "--decode-attention-backend", "fa4", "--speculative-algorithm", "DSPARK",
               "--speculative-draft-attention-backend", "flashinfer",
               "--speculative-num-draft-tokens", "8", "--disable-cuda-graph"]
        env = {"PYTHONPATH": "/repo/python", "SGLANG_DSPARK_PREFIX_CACHE": "1"}
        return cmd, env

    def test_profile_command_is_explicit_and_does_not_mutate_input(self):
        cmd, env = self.command()
        result, changed = profile_command(cmd, env, ordinary=False, profile_id="profile",
            runtime_folder="/runtime", hook_folder="/hooks")
        self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "flashinfer")
        self.assertNotIn("SGLANG_DSPARK_PREFIX_CACHE_FINGERPRINT", env)
        for name in ("--attention-backend", "--prefill-attention-backend", "--decode-attention-backend",
                     "--speculative-draft-attention-backend"):
            self.assertEqual(result[result.index(name) + 1], "fa4")
        self.assertIn("--enable-deterministic-inference", result)
        self.assertNotIn("--disable-radix-cache", result)
        self.assertEqual(changed["SGLANG_DSPARK_PREFIX_CACHE"], "1")
        self.assertEqual(changed["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"], "0")
        self.assertEqual(changed["SGLANG_DSPARK_PREFIX_CACHE_FINGERPRINT"], "profile")

    def test_ordinary_target_has_no_speculative_args_and_cache_off(self):
        result, _ = profile_command(*self.command(), ordinary=True, profile_id="profile",
            runtime_folder="/runtime", hook_folder="/hooks")
        self.assertFalse(any(value.startswith("--speculative-") for value in result))
        self.assertNotIn("DSPARK", result)
        self.assertIn("--disable-radix-cache", result)

    def valid_info(self):
        return dict(enable_deterministic_inference=True, disable_cuda_graph=True, disable_radix_cache=False,
            attention_backend="fa4", prefill_attention_backend="fa4", decode_attention_backend="fa4",
            dtype="bfloat16", kv_cache_dtype="bfloat16", bf16_gemm_backend="torch", tp_size=1,
            page_size=1, quantization=None, speculative_algorithm="DSPARK", speculative_draft_attention_backend="fa4")

    def test_resolved_profile_must_not_fall_back_to_flashinfer_or_disable_cache(self):
        info = self.valid_info()
        validate_profile(info, False)
        validate_profile({**info, "disable_radix_cache": True}, True)
        for name, value in (("disable_radix_cache", True), ("attention_backend", "flashinfer"),
                            ("prefill_attention_backend", "flashinfer"), ("disable_cuda_graph", False),
                            ("enable_deterministic_inference", False), ("kv_cache_dtype", "fp8_e4m3"),
                            ("speculative_draft_attention_backend", "flashinfer")):
            with self.subTest(name=name), self.assertRaises(ab.SafetyFailure):
                validate_profile({**info, name: value}, False)

    def test_acceptance_required_for_dspark_not_for_independent_target(self):
        row = dict(cell="cell", variant=0, phase="cold_reference", cached_tokens=0, prompt_tokens=128,
                   output_ids=[1] * 32, meta_info={})
        spec = {"prompt_length": 128}
        validate_row(row, spec, 32, ordinary=True, require_cold=True)
        with self.assertRaises(ab.SafetyFailure):
            validate_row(row, spec, 32, ordinary=False, require_cold=True)
        row["meta_info"] = {key: 0 for key in ab.ACCEPTANCE_KEYS}
        validate_row(row, spec, 32, ordinary=False, require_cold=True)
        with self.assertRaises(ab.SafetyFailure):
            validate_row({**row, "cached_tokens": 1}, spec, 32, ordinary=False, require_cold=True)

    def test_runtime_attestation_fails_closed(self):
        data = dict(torch_batch_invariant=True, fa4_batch_invariant=True, backend_class="FlashAttentionBackend",
            fa_impl_ver=4, num_splits=1, decode_num_splits=1, page_size=1, kv_dtype="torch.bfloat16",
            deterministic=True, graphs_disabled=True, gemm_backend="torch")
        validate_runtime(data)
        for name, value in (("torch_batch_invariant", False), ("fa4_batch_invariant", False),
                            ("num_splits", 0), ("decode_num_splits", 0), ("kv_dtype", "torch.float16")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_runtime({**data, name: value})


if __name__ == "__main__":
    unittest.main()
