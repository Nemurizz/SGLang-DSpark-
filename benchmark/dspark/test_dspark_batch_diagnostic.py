"""CPU tests of diagnostic isolation, never starts the model."""
from pathlib import Path
import json
import tempfile
import unittest

import dspark_batch_diagnostic as diag


class TestDiagnostic(unittest.TestCase):
    def args(self, profile):
        return diag.parse_args(["--repo", "/repo", "--model", "/model", "--draft", "/draft",
                               "--sps-table", "/sps.json", "--dataset", "/data.json",
                               "--output", "/new", "--profile", profile])

    def test_target_removes_all_speculative_args(self):
        cmd, env = diag.DiagnosticServer(self.args("target-fi"), "full", Path("/new")).command()
        self.assertFalse(any(x.startswith("--speculative-") for x in cmd))
        for flag in ("--disable-cuda-graph", "--disable-radix-cache", "--enable-deterministic-inference"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "flashinfer")
        self.assertEqual(env["SGLANG_DSPARK_PREFIX_CACHE"], "0")

    def test_fa4_only_changes_target_backend(self):
        cmd, _ = diag.DiagnosticServer(self.args("dspark-fa4"), "full", Path("/new")).command()
        self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "fa4")
        self.assertEqual(cmd[cmd.index("--speculative-draft-attention-backend") + 1], "flashinfer")
        self.assertEqual(cmd[cmd.index("--speculative-algorithm") + 1], "DSPARK")

    def test_builtin_triton_attention_keeps_existing_other_backends(self):
        cmd, _ = diag.DiagnosticServer(self.args("dspark-triton"), "full", Path("/new")).command()
        self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "triton")
        self.assertEqual(cmd[cmd.index("--decode-attention-backend") + 1], "fa4")
        self.assertEqual(cmd[cmd.index("--speculative-draft-attention-backend") + 1], "flashinfer")

    def test_target_comparison_does_not_require_spec_metadata(self):
        reference = {"output_ids": [1, 2], "meta_info": {}}
        row = {"output_ids": [1, 2], "meta_info": {}, "cell": "p2048_c8", "variant": 3, "phase": "concurrent"}
        diag.ab.check_reference(reference, row, check_acceptance=False)
        row["output_ids"] = [1, 3]
        with self.assertRaises(diag.ab.SafetyFailure):
            diag.ab.check_reference(reference, row, check_acceptance=False)

    def test_optional_logprobs_preserve_sampling_and_input(self):
        args = self.args("dspark-fi")
        body = diag.diagnostic_body([2, 3], args)
        self.assertNotIn("return_logprob", body)
        args.logprob_topk = 3
        enriched = diag.diagnostic_body([2, 3], args)
        self.assertEqual(body["sampling_params"], enriched["sampling_params"])
        self.assertEqual(enriched["input_ids"], [2, 3])
        self.assertTrue(enriched["return_logprob"])
        self.assertEqual(enriched["top_logprobs_num"], 3)

    def test_independent_oracle_must_cover_every_variant_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oracle.jsonl"
            row = {"cell": "p2048_c8", "variant": 0, "phase": "reference", "output_ids": [1]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(diag.load_oracle(path, "p2048_c8", 1)[0]["output_ids"], [1])
            with self.assertRaises(ValueError):
                diag.load_oracle(path, "p2048_c8", 8)
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_oracle(path, "p2048_c8", 1)


if __name__ == "__main__":
    unittest.main()
