import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import dspark_prefix_ab as ab
from cache_profile_correctness import publish_complete, validate_profile, validate_row


class TestOriginalProfileGate(unittest.TestCase):
    def test_only_original_profile_can_be_measured(self):
        args = SimpleNamespace(deterministic_control_only=False, reference_verify_plans=False,
                               enable_standard_graphs=False)
        info = dict(enable_deterministic_inference=False, disable_cuda_graph=True, disable_radix_cache=False)
        validate_profile(args, info)
        for flag in vars(args):
            with self.assertRaises(ValueError):
                validate_profile(SimpleNamespace(**{**vars(args), flag: True}), info)
        for flag in info:
            with self.assertRaises(ab.SafetyFailure):
                validate_profile(args, {**info, flag: not info[flag]})

    def test_incomplete_matrix_cannot_publish_oracle(self):
        dataset = {"cell": {"concurrency": 2}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for refs, completed in (({}, 0), ({("cell", 0): {}}, 1),
                                    ({("cell", 0): {}, ("cell", 1): {}}, 0)):
                with self.assertRaises(ValueError):
                    publish_complete(root, dataset, refs, completed)
                self.assertFalse((root / "references.jsonl").exists())

    def test_complete_oracle_is_unique_and_never_overwritten(self):
        dataset = {"cell": {"concurrency": 2}}
        refs = {("cell", i): {"cell": "cell", "variant": i, "phase": "reference"} for i in range(2)}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(len(publish_complete(root, dataset, refs, 1)), 64)
            rows = [json.loads(line) for line in (root / "references.jsonl").read_text().splitlines()]
            self.assertEqual([r["variant"] for r in rows], [0, 1])
            with self.assertRaises(ValueError):
                publish_complete(root, dataset, refs, 1)

    def test_cache_reuse_wrong_lengths_or_missing_acceptance_stop(self):
        row = dict(cached_tokens=0, prompt_tokens=128, completion_tokens=32, output_ids=[1] * 32,
                   meta_info={key: 0 for key in ab.ACCEPTANCE_KEYS})
        spec = {"prompt_length": 128}
        validate_row(row, spec, 32)
        for key, value in (("cached_tokens", 1), ("prompt_tokens", 129), ("completion_tokens", 31),
                           ("meta_info", {}), ("output_ids", [1])):
            with self.assertRaises(ab.SafetyFailure):
                validate_row({**row, key: value}, spec, 32)


if __name__ == "__main__":
    unittest.main()
