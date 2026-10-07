import json
from pathlib import Path
import tempfile
import unittest

from trace_target_verify import config, load_expected


class TestTargetVerifyTrace(unittest.TestCase):
    def test_focus_uses_first_committed_anchor_not_mismatching_future_token(self):
        cfg = config(Path('/out'), [1, 2, 3], {'output_ids': [576, 2197, 34334]})
        self.assertEqual(cfg['position'], 3)
        self.assertEqual(cfg['anchor_token'], 576)
        self.assertTrue(cfg['observe_layer0_projection'])
        self.assertEqual(cfg['max_captures'], 10)

    def test_expected_must_be_unique_preserved_cold_control(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path = folder / 'requests.jsonl'
            control = dict(cell='p128_c1', variant=0, phase='cold_reference', cached_tokens=0)
            for rows in ([], [{**control, 'cached_tokens': 1}], [control, control]):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    load_expected(folder)
            path.write_text(json.dumps(control) + '\n')
            self.assertEqual(load_expected(folder), control)


if __name__ == '__main__':
    unittest.main()
