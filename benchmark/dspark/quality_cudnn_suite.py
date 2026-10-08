"""Independent cuDNN route; retain/reuse successful immutable BF16 sentinels.

No retry of h64 CUTLASS failure. Quality data, grading and time limits unchanged.
Automatically continue into paired scores/ABBA only after execution gates pass.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from quality_data import sha
from quality_report import report
from quant_feasibility_gate import baseline_identity, launch_args_from_template

TEMPLATE = Path('/root/autodl-fs/dspark-verify-all-ab-20261008-h47/manifest.json')
CHECKPOINT = Path('/root/autodl-tmp/models/Qwen3-14B-NVFP4-20261008-h63')
DATA_SHA = 'c4ba785e66d5535c7d454a25c19af692aab686c0ea94df204a12b0f6a649488f'


def run(root, previous, recipe='nvfp4'):
    if (root / 'suite_status.json').exists():
        raise ValueError('Fresh execution evidence required')
    smoke = json.loads((root / 'backend_smoke.json').read_text())
    valid_backend = smoke.get('backend') == 'cudnn' if recipe == 'nvfp4' else smoke.get('recipe') == 'w8a8_fp8_native'
    if smoke['status'] != 'passed_backend_smoke_only' or not valid_backend or len(smoke['cases']) != 16:
        raise ValueError('Missing independently qualified backend preflight')
    backend = 'flashinfer_cudnn' if recipe == 'nvfp4' else 'flashinfer_cutlass'
    preflight = json.loads((previous / 'grader_preflight.json').read_text())
    if not preflight['passed'] or len(preflight['canonical_mbpp']) != 64:
        raise ValueError('Missing preserved grader preflight')
    dataset = previous / 'data-v3/dataset.json'
    if sha(dataset) != DATA_SHA:
        raise ValueError('Immutable dataset changed')
    identity = baseline_identity(launch_args_from_template(json.loads(TEMPLATE.read_text())).repo)
    reused = {}
    for arm in ('bf16-ar-sentinels', 'bf16-dspark-sentinels'):
        folder = previous / arm
        status = json.loads((folder / 'status.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        if (status['status'] != 'passed' or status['completed_requests'] != 56 or
                manifest['dataset_sha256'] != DATA_SHA or manifest['baseline_before'] != identity or
                status['baseline_after'] != identity or manifest['quant']):
            raise ValueError('BF16 sentinel evidence not reusable')
        reused[arm] = {'folder': str(folder), 'requests_sha256': sha(folder / 'requests.jsonl'),
                      'manifest_sha256': sha(folder / 'manifest.json'), 'status_sha256': sha(folder / 'status.json')}
    common = [sys.executable, '-u', str(Path(__file__).with_name('quality_run.py')),
        '--template', str(TEMPLATE), '--checkpoint', str(CHECKPOINT), '--dataset', str(dataset),
        '--jail', str(previous / 'jail-v3'), '--fp4-backend', backend, '--recipe', recipe]
    plan = [('quant-ar-sentinels', 'quant-ar', True, None),
            ('quant-dspark-sentinels', 'quant-dspark', True, 'quant-ar-sentinels'),
            ('bf16-quality', 'bf16-dspark', False, None),
            ('quant-quality', 'quant-dspark', False, None)]
    state = {'status': 'running', 'plan': plan, 'completed': [], 'reused_bf16_gates': reused,
             'dataset_sha256': DATA_SHA, 'started_at': time.time(), 'recipe': recipe,
             'backend': 'flashinfer_cudnn' if recipe == 'nvfp4' else 'sgl_kernel_cutlass_fp8'}
    path = root / 'suite_status.json'
    def save():
        path.write_text(json.dumps(state, indent=2))
    for name, arm, sentinels, reference in plan:
        state['active'] = name
        save()
        cmd = common + ['--arm', arm, '--output', str(root / name)]
        if sentinels:
            cmd.append('--sentinels')
        if reference:
            cmd += ['--references', str(root / reference / 'requests.jsonl')]
        with (root / (name + '.log')).open('x') as out:
            code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
        if code:
            state.update(status='execution_safety_stopped', exit_code=code, finished_at=time.time())
            save()
            print(json.dumps(state), flush=True)
            return code
        state['completed'].append(name)
        print('COMPLETED ' + name, flush=True)
    state.update(status='quality_complete_NOT_performance_proof', active=None)
    save()
    comparison = report(root, root / 'quality_comparison.json')
    if comparison['quality_gate'] != 'passed_for_fixed_subsets_only':
        state.update(status='quality_budget_failed_candidate_not_timed', finished_at=time.time())
        save()
        return 3
    state.update(status='steady_ab_running', active='steady_ab')
    save()
    cmd = [sys.executable, '-u', str(Path(__file__).with_name('quant_steady_ab.py')),
           '--quality', str(root / 'quality_comparison.json'), '--template', str(TEMPLATE),
           '--checkpoint', str(CHECKPOINT), '--output', str(root / 'steady_ab'),
           '--fp4-backend', backend, '--recipe', recipe]
    with (root / 'steady_ab.log').open('x') as out:
        code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    state.update(status='qualified_fixed_subsets_and_one_cell' if code == 0 else 'benefit_not_proven',
                 exit_code=code, finished_at=time.time())
    save()
    print(json.dumps(state), flush=True)
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(run(args.root, args.previous))
