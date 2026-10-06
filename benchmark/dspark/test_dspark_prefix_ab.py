"""CPU-only protocol tests; does not import torch/transformers or start a server."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("dspark_prefix_ab", Path(__file__).with_name("dspark_prefix_ab.py"))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == " ":
            return [32]
        return [int.from_bytes(word.encode(), "little") for word in text.split()]


class TestDSparkAB(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(ab.percentile([0, 10], .95), 9.5)
        self.assertIsNone(ab.percentile([], .95))

    def test_sse_fragmented_multibyte_and_done(self):
        wire = ('data: {"text":"中文"}\r\n\r\ndata: [DONE]\n\n').encode()

        async def collect():
            return [f async for f in ab.sse_frames(FakeContent([wire[i:i+1] for i in range(len(wire))]))]

        self.assertEqual(asyncio.run(collect()), ['{"text":"中文"}', '[DONE]'])

    def test_stream_metadata_does_not_start_ttft(self):
        tracker = ab.StreamTracker(10)
        tracker.add({"output_ids": [], "meta_info": {}}, 10.1)
        tracker.add({"output_ids": [1, 2], "meta_info": {}}, 10.2)
        tracker.add({"output_ids": [1, 2, 3, 4], "meta_info": {
            "finish_reason": {"type": "length"}, "completion_tokens": 4, "prompt_tokens": 128}}, 10.5)
        row = tracker.result(10.6)
        self.assertAlmostEqual(row["ttft_ms"], 200)
        self.assertAlmostEqual(row["tpot_ms"], 100)
        self.assertAlmostEqual(row["e2e_ms"], 600)
        self.assertEqual([e["new_tokens"] for e in row["stream_events"]], [2, 2])

    def test_stream_rejects_rewritten_tokens(self):
        tracker = ab.StreamTracker(0)
        tracker.add({"output_ids": [1, 2]}, .1)
        with self.assertRaises(ab.SafetyFailure):
            tracker.add({"output_ids": [1, 4, 5]}, .2)

    def test_stream_rejects_text_only(self):
        with self.assertRaises(ab.SafetyFailure):
            ab.StreamTracker(0).add({"text": "abc"}, .1)

    def test_incomplete_stream_is_failure(self):
        tracker = ab.StreamTracker(0)
        tracker.add({"output_ids": [1]}, .1)
        with self.assertRaises(ab.SafetyFailure):
            tracker.result(.2)

    def test_counter_delta_ignores_gauges(self):
        before = {k: 10 for k in ab.COUNTERS}
        after = {k: 12 for k in ab.COUNTERS}
        before["ready_tokens"], after["ready_tokens"] = 100, 1
        self.assertEqual(ab.counter_delta(before, after)["match_calls"], 2)
        self.assertNotIn("ready_tokens", ab.counter_delta(before, after))

    def test_counter_reset_is_failure(self):
        with self.assertRaises(ab.SafetyFailure):
            ab.counter_delta({"match_calls": 5}, {"match_calls": 0})

    def test_shadow_token_rates_and_means(self):
        delta = {"match_calls": 2, "target_hit_tokens": 20, "paired_hit_tokens": 10}
        rows = [{"prompt_tokens": 16, "cached_tokens": 10}] * 2
        result = ab.cache_metrics(delta, rows, True)
        self.assertAlmostEqual(result["observed_paired_token_hit_rate"], 1/3)
        self.assertEqual(result["paired_over_target_token_coverage"], .5)
        self.assertEqual(result["paired_hit_length_mean_per_match"], 5)

    def test_retried_match_invalidates_request_token_denominator(self):
        result = ab.cache_metrics({"match_calls": 3, "target_hit_tokens": 20, "paired_hit_tokens": 10},
                                  [{"prompt_tokens": 16, "cached_tokens": 10}] * 2, True)
        self.assertIsNone(result["observed_paired_token_hit_rate"])
        self.assertFalse(result["token_rate_denominator_valid"])

    def test_disabled_sidecar_is_not_reported_as_zero_hit_rate(self):
        result = ab.cache_metrics({"match_calls": 0, "target_hit_tokens": 0, "paired_hit_tokens": 0},
                                  [{"prompt_tokens": 128, "cached_tokens": 0}], False)
        self.assertIsNone(result["observed_paired_token_hit_rate"])
        self.assertEqual(result["actual_cached_token_rate"], 0)

    def test_reference_gate_token_and_acceptance(self):
        row = {"cell": "p128_c1", "variant": 0, "phase": "steady", "output_ids": [1, 2],
               "meta_info": {k: [0, 1] if "histogram" in k else 1 for k in ab.ACCEPTANCE_KEYS}}
        ab.check_reference(row, row)
        changed = copy.deepcopy(row)
        changed["output_ids"][1] = 3
        with self.assertRaisesRegex(ab.SafetyFailure, "token mismatch"):
            ab.check_reference(row, changed)
        changed = copy.deepcopy(row)
        changed["meta_info"]["spec_verify_ct"] = 2
        with self.assertRaisesRegex(ab.SafetyFailure, "acceptance changed"):
            ab.check_reference(row, changed)

    def test_dataset_exact_lengths_reproducible_variant_diversity(self):
        data = ab.make_dataset(FakeTokenizer(), 9)
        self.assertEqual(data, ab.make_dataset(FakeTokenizer(), 9))
        self.assertEqual(len(data), 12)
        for spec in data.values():
            self.assertEqual(len(spec["prompts"]), spec["concurrency"])
            self.assertTrue(all(len(p) == spec["prompt_length"] for p in spec["prompts"]))
            self.assertEqual(len({tuple(p) for p in spec["prompts"]}), spec["concurrency"])

    def test_p95_gate_insufficient_and_regression(self):
        base = {"request_count": 8, "cell": "p128_c1", **{k: {"p95": 10} for k in ("ttft_ms", "tpot_ms", "e2e_ms")}}
        bad = copy.deepcopy(base)
        bad["e2e_ms"]["p95"] = 11
        self.assertEqual(ab.check_p95(base, bad, 64, 0), "insufficient_samples")
        with self.assertRaisesRegex(ab.SafetyFailure, "p95 regression"):
            ab.check_p95(base, bad, 8, 0)
        self.assertEqual(ab.check_p95(base, base, 8, 0), "passed")

    def test_unsafe_mode_is_explicit_and_keeps_dspark_guard(self):
        args = ab.parse_args(["--model", "target", "--draft", "draft", "--output", "new", "--repo", "/repo"])
        _, env = ab.ManagedServer(args, "target_only_shadow", Path("unused")).command()
        self.assertEqual(env["SGLANG_DSPARK_DISABLE_RADIX_CACHE"], "1")
        self.assertEqual(env["SGLANG_DSPARK_PREFIX_CACHE_SHADOW"], "1")
        self.assertEqual(env["SGLANG_DSPARK_RAGGED_VERIFY_GRAPH"], "0")

    def test_smoke_protocol_defaults(self):
        args = ab.parse_args(["--model", "target", "--draft", "draft", "--output", "new", "--preset", "smoke"])
        self.assertEqual((args.requests, args.repeats, args.output_tokens), (8, 1, 32))
        self.assertFalse(args.include_target_only_shadow)
        self.assertFalse(args.enable_standard_graphs)

    def test_no_missing_metrics_coerced_to_zero(self):
        self.assertIsNone(ab.counter_delta({}, {})["partial_recovery_calls"])

    def test_sampling_seed_matches_native_generate_schema(self):
        body = ab.request_body([1, 2], 32, 7)
        self.assertEqual(body["sampling_params"]["sampling_seed"], 7)
        self.assertNotIn("seed", body["sampling_params"])


if __name__ == "__main__":
    unittest.main()
