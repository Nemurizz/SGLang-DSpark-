import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prepare_nvfp4 import REPO, REVISION
from stage_verified_checkpoint import main


class CheckpointStagingTests(unittest.TestCase):
    def run_stage(self, manifest_update=None, *, corrupt=False, existing=False):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / 'source', root / 'destination'
            (source / 'model').mkdir(parents=True)
            data = b'checkpoint fixture'
            (source / 'model' / 'config.json').write_bytes(data)
            manifest = dict(status='verified', repo=REPO, revision=REVISION,
                            files=[dict(name='config.json', bytes=len(data),
                                        sha256='0'*64 if corrupt else hashlib.sha256(data).hexdigest())])
            manifest.update(manifest_update or {})
            (source / 'manifest.json').write_text(json.dumps(manifest))
            if existing:
                destination.mkdir()
                (destination / 'preserve').write_text('original')
            with patch('sys.argv', ['stage', '--source', str(source), '--destination', str(destination)]):
                if corrupt or existing or manifest_update:
                    with self.assertRaises(ValueError):
                        main()
                    self.assertFalse((destination / 'manifest.json').exists())
                else:
                    main()
                    self.assertEqual((destination / 'model' / 'config.json').read_bytes(), data)
                    self.assertEqual((destination / 'manifest.json').read_bytes(),
                                     (source / 'manifest.json').read_bytes())
            self.assertEqual((source / 'model' / 'config.json').read_bytes(), data)
            if existing:
                self.assertEqual((destination / 'preserve').read_text(), 'original')

    def test_copy_and_publish_manifest_last(self):
        self.run_stage()

    def test_bad_copy_hash_cannot_publish_verified_manifest(self):
        self.run_stage(corrupt=True)

    def test_existing_destination_is_preserved(self):
        self.run_stage(existing=True)

    def test_unverified_source_is_refused(self):
        self.run_stage({'status': 'downloading'})


if __name__ == '__main__':
    unittest.main()
