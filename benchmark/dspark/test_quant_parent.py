import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from audit_nvfp4_parent import tensor_sha


class ParentAuditTests(unittest.TestCase):
    def checkpoint(self, root, bounds):
        data = b'12345678'
        header = json.dumps({'lm_head.weight': {
            'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': bounds}}).encode()
        (root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'lm_head.weight': 'model.safetensors'}}))
        (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + data)
        return data

    def test_sha_covers_only_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.checkpoint(root, [0, 8])
            self.assertEqual(tensor_sha(root, 'lm_head.weight'), {
                'dtype': 'BF16', 'shape': [2, 2], 'bytes': 8,
                'sha256': hashlib.sha256(data).hexdigest()})

    def test_truncated_and_negative_byte_bounds_fail_closed(self):
        for bounds in ([0, 32], [-1, 8], [8, 8]):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.checkpoint(root, bounds)
                with self.assertRaises(ValueError):
                    tensor_sha(root, 'lm_head.weight')


if __name__ == '__main__':
    unittest.main()
