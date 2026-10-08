"""Download a pinned, public NVIDIA checkpoint; never alter baseline weights.

No quantization/training/dependency upgrades. Mirror bytes must match official
LFS digests, and the full verified manifest is required before inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

REPO = 'nvidia/Qwen3-14B-NVFP4'
REVISION = 'bc39319a4dc265d9bbb9a9731bc52c4988d9ece7'
OFFICIAL_LFS = {
    'model-00001-of-00003.safetensors': '3cea84e60aac95e5ef75166c50cd9ddf8401c267b0c13a3b426990cdc72a09bf',
    'model-00002-of-00003.safetensors': '43fbfd28d5032cd2940a82875ce3d2cc3c3f3cc9d8872c4c26c5ca3fe8001f76',
    'model-00003-of-00003.safetensors': 'b4bf668aa6f8535dd467a9a3339116b536682b4241972054b783d514cbe84e50',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    a = p.parse_args()
    root = a.root.resolve()
    if root.exists():
        raise ValueError('Use a new output directory; partial downloads are preserved')
    root.mkdir(parents=True)
    os.environ.update(HF_ENDPOINT=a.endpoint, HF_HOME=str(root / 'hf-cache'),
                      HF_XET_CACHE=str(root / 'xet-cache'), HF_HUB_DISABLE_XET='1',
                      HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='30',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(endpoint=a.endpoint, token=False).model_info(
        REPO, revision=REVISION, files_metadata=True, timeout=30)
    if info.sha != REVISION or info.gated or info.private:
        raise ValueError('Unexpected model identity/access')
    files = [f for f in info.siblings if f.rfilename.endswith('.safetensors') or
             f.rfilename in ('config.json', 'hf_quant_config.json',
                              'model.safetensors.index.json', 'README.md')]
    total = sum(f.size for f in files)
    if shutil.disk_usage(root).free < total + 2 * 1024 ** 3:
        raise ValueError('Insufficient destination space; do not delete baseline files')
    manifest = {'repo': REPO, 'revision': REVISION, 'endpoint': a.endpoint,
                'expected_bytes': total, 'status': 'downloading', 'files': []}
    for f in files:
        lfs = f.lfs
        if f.rfilename.endswith('.safetensors') and getattr(lfs, 'sha256', None) != OFFICIAL_LFS.get(f.rfilename):
            raise ValueError('Mirror metadata differs from pinned official LFS digest')
        manifest['files'].append({'name': f.rfilename, 'bytes': f.size,
            'lfs_sha256': getattr(lfs, 'sha256', None), 'git_blob_sha1': f.blob_id})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'revision': REVISION, 'expected_bytes': total}), flush=True)
    model = root / 'model'
    snapshot_download(REPO, revision=REVISION, endpoint=a.endpoint, token=False,
                      local_dir=model, allow_patterns=[f.rfilename for f in files], max_workers=2)
    for f in manifest['files']:
        path = model / f['name']
        if path.stat().st_size != f['bytes']:
            raise ValueError('File size mismatch: ' + f['name'])
        actual = digest(path)
        if f['lfs_sha256']:
            if actual != f['lfs_sha256']:
                raise ValueError('LFS checksum mismatch: ' + f['name'])
        else:
            b = path.read_bytes()
            git_sha = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
            if git_sha != f['git_blob_sha1']:
                raise ValueError('Git blob checksum mismatch: ' + f['name'])
        f['sha256'] = actual
        print('VERIFIED ' + f['name'], flush=True)
    q = json.loads((model / 'hf_quant_config.json').read_text())['quantization']
    if q['quant_algo'] != 'NVFP4' or q['group_size'] != 16 or 'lm_head' not in q['exclude_modules']:
        raise ValueError('Unqualified quantization recipe')
    manifest.update(status='verified', quantization=q,
                    kv_runtime_override='bfloat16', draft_runtime_override='unquant')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PREPARED ' + str(model), flush=True)


if __name__ == '__main__':
    main()
