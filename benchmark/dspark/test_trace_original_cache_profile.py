import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import dspark_prefix_ab as ab
from trace_original_cache_profile import TraceServer, trace_config, validate_server_info


class TestProfileIsolation(unittest.TestCase):
    def test_only_committed_first_anchor_selected(self):
        config = trace_config(Path("run"), [1, 2, 3], {"output_ids": [7, 8]})
        self.assertEqual(config["position"], 3)
        self.assertEqual(config["anchor_token"], 7)
        self.assertEqual(config["max_captures"], 10)

    def test_original_radix_structure_graph_and_profile_settings(self):
        valid = dict(enable_deterministic_inference=True, disable_cuda_graph=True, disable_radix_cache=True)
        validate_server_info(valid, True)
        validate_server_info({**valid, "enable_deterministic_inference": False,
                              "disable_radix_cache": False}, False)
        for key in valid:
            with self.assertRaises(ab.SafetyFailure):
                validate_server_info({**valid, key: False}, True)

    def test_command_keeps_candidate_off_and_original_profiles(self):
        args = SimpleNamespace(repo="repo", model="model", draft="draft", port=30117,
            mem_fraction=.55, seed=20261007, cache_budget_fraction=.25, sps_table=None,
            enable_standard_graphs=False, deterministic_control_only=True, reference_verify_plans=False)
        cmd, env = TraceServer(args, Path("run"), {}).command()
        self.assertIn("--enable-deterministic-inference", cmd)
        self.assertIn("--disable-cuda-graph", cmd)
        self.assertEqual(env["SGLANG_DSPARK_VERIFY_SERIAL_REFERENCE"], "0")
        self.assertEqual(env["SGLANG_DSPARK_PREFIX_CACHE"], "0")
        args.deterministic_control_only = False
        cmd, _ = TraceServer(args, Path("run"), {}).command()
        self.assertNotIn("--enable-deterministic-inference", cmd)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required for CPU snapshot checks")
class TestProfileComparison(unittest.TestCase):
    def test_bitwise_difference_and_dtype_guard(self):
        import torch
        from compare_original_cache_profiles import differences
        left = torch.tensor([1., 2.], dtype=torch.bfloat16)
        self.assertTrue(differences(left, left.clone())["bitwise_equal"])
        self.assertEqual(differences(left, left + 1)["different_elements"], 2)
        with self.assertRaises(ValueError):
            differences(left, left.float())

    def test_compared_anchors_require_same_identity(self):
        import torch
        from compare_original_cache_profiles import compare
        meta = dict(prompt_sha256="x", prompt_length=128, position=128,
                    anchor_token=576, stage="logits", batch_size=1)
        tensors = {"x": torch.tensor([1.])}
        self.assertTrue(compare((meta, tensors), (meta, tensors))["x"]["bitwise_equal"])
        with self.assertRaises(ValueError):
            compare((meta, tensors), ({**meta, "anchor_token": 7}, tensors))
        with self.assertRaises(ValueError):
            compare((meta, tensors), (meta, {}))

    def test_incomplete_trace_is_not_interpreted(self):
        from compare_original_cache_profiles import load
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "status.json").write_text(json.dumps({"status": "safety_stopped"}))
            with self.assertRaises(ValueError):
                load(path)


if __name__ == "__main__":
    unittest.main()
