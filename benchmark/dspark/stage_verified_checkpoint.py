"""Copy a verified, pinned checkpoint to fresh local storage; preserve source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_nvfp4 import REPO, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    args = p.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    if (manifest['status'], manifest['repo'], manifest['revision']) != ('verified', REPO, REVISION):
        raise ValueError('Unverified or unexpected source checkpoint')
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Use a fresh, separate destination; never overwrite evidence')
    total = sum(f['bytes'] for f in manifest['files'])
    if shutil.disk_usage(destination.parent).free < total + 3 * 1024**3:
        raise ValueError('Insufficient local disk space including serving headroom')
    (destination / 'model').mkdir(parents=True)
    for entry in manifest['files']:
        name = entry['name']
        if Path(name).name != name:
            raise ValueError('Checkpoint entry is not a simple filename')
        src, dst = source / 'model' / name, destination / 'model' / name
        if src.stat().st_size != entry['bytes']:
            raise ValueError('Source size changed: ' + name)
        shutil.copy2(src, dst)
        digest = hashlib.sha256()
        with dst.open('rb') as stream:
            while block := stream.read(16 * 1024 * 1024):
                digest.update(block)
        if dst.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Local copy verification failed: ' + name)
        print('VERIFIED LOCAL ' + name, flush=True)
    # Publish the verified manifest last; failed copies cannot enter the gate.
    shutil.copy2(source / 'manifest.json', destination / 'manifest.json')
    print('STAGED ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
