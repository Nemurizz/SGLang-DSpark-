"""Sequential scoped services; stop on first execution safety failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(root):
    preflight = json.loads((root / 'grader_preflight.json').read_text())
    if not preflight['passed'] or len(preflight['canonical_mbpp']) != 64:
        raise ValueError('Require successful canonical grader/isolation preflight')
    common = [sys.executable, '-u', str(Path(__file__).with_name('quality_run.py')),
        '--template', '/root/autodl-fs/dspark-verify-all-ab-20261008-h47/manifest.json',
        '--checkpoint', '/root/autodl-tmp/models/Qwen3-14B-NVFP4-20261008-h63',
        '--dataset', str(root / 'data-v3/dataset.json'), '--jail', str(root / 'jail-v3')]
    plan = [('bf16-ar-sentinels', 'bf16-ar', True, None),
            ('bf16-dspark-sentinels', 'bf16-dspark', True, 'bf16-ar-sentinels'),
            ('quant-ar-sentinels', 'quant-ar', True, None),
            ('quant-dspark-sentinels', 'quant-dspark', True, 'quant-ar-sentinels'),
            ('bf16-quality', 'bf16-dspark', False, None),
            ('quant-quality', 'quant-dspark', False, None)]
    status_path = root / 'suite_status.json'
    if status_path.exists():
        raise ValueError('Do not overwrite/retry suite evidence')
    status = {'status': 'running', 'plan': plan, 'completed': [], 'started_at': time.time()}
    for name, arm, sentinels, reference in plan:
        status['active'] = name
        status_path.write_text(json.dumps(status, indent=2))
        cmd = common + ['--arm', arm, '--output', str(root / name)]
        if sentinels:
            cmd += ['--sentinels']
        if reference:
            cmd += ['--references', str(root / reference / 'requests.jsonl')]
        with (root / (name + '.log')).open('x') as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='safety_stopped', exit_code=result.returncode, finished_at=time.time())
            status_path.write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            return result.returncode
        status['completed'].append(name)
        print('COMPLETED ' + name, flush=True)
    status.update(status='quality_complete_NOT_performance_proof', active=None, finished_at=time.time())
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    raise SystemExit(run(p.parse_args().root))
