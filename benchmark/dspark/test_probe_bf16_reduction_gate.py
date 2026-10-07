import json
from pathlib import Path
import tempfile
import unittest

import dspark_prefix_ab as ab
from probe_bf16_reduction_gate import original_controls
from cache_profile_correctness import ORIGINAL


class TestFrozenControls(unittest.TestCase):
    def write_controls(self, folder, change=None):
        dataset = {cell: {"prompt_length": 128, "concurrency": concurrency}
                   for cell, concurrency in (("p128_c1", 1), ("p128_c4", 4))}
        rows = [dict(cell=cell, variant=v, phase="reference", original_revision=ORIGINAL,
                     cached_tokens=0, prompt_tokens=128, completion_tokens=32, output_ids=[1] * 32,
                     meta_info={key: 0 for key in ab.ACCEPTANCE_KEYS})
                for cell, spec in dataset.items() for v in range(spec["concurrency"])]
        if change:
            change(rows)
        (folder / "dataset.json").write_text(json.dumps(dataset))
        (folder / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    def test_valid_original_controls_read_only_and_no_new_oracle(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.write_controls(folder)
            before = {path.name: path.read_bytes() for path in folder.iterdir()}
            _, refs = original_controls(folder)
            self.assertEqual(len(refs), 5)
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})

    def test_candidate_reference_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.write_controls(folder, lambda rows: rows[0].update(original_revision="candidate"))
            with self.assertRaises(ValueError):
                original_controls(folder)

    def test_incomplete_controls_are_refused(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.write_controls(folder, lambda rows: rows.pop())
            with self.assertRaises(ValueError):
                original_controls(folder)

    def test_cache_reuse_and_missing_acceptance_are_refused(self):
        for change in (lambda rows: rows[0].update(cached_tokens=1),
                       lambda rows: rows[0].update(meta_info={})):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                folder = Path(name)
                self.write_controls(folder, change)
                with self.assertRaises(ab.SafetyFailure):
                    original_controls(folder)


if __name__ == "__main__":
    unittest.main()
