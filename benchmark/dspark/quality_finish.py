"""Wait for the owned suite, grade it, and launch ABBA only on quality pass."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from quality_report import report


def finish(root):
    output = root / 'qualification_status.json'
    if output.exists():
        raise ValueError('Fresh qualification output required; no retries')
    status = {'status': 'waiting_for_quality', 'started_at': time.time()}
    output.write_text(json.dumps(status, indent=2))
    while True:
        suite = json.loads((root / 'suite_status.json').read_text())
        if suite['status'] != 'running':
            break
        time.sleep(10)
    if suite['status'] != 'quality_complete_NOT_performance_proof':
        status.update(status='execution_safety_stopped', suite_status=suite)
    else:
        result = report(root, root / 'quality_comparison.json')
        status['quality_gate'] = result['quality_gate']
        if result['quality_gate'] != 'passed_for_fixed_subsets_only':
            status.update(status='quality_budget_failed_candidate_not_timed',
                          next_step='inspect discordant cases; independently qualify conservative precision')
        else:
            status['status'] = 'steady_ab_running'
            output.write_text(json.dumps(status, indent=2))
            command = [sys.executable, '-u', str(Path(__file__).with_name('quant_steady_ab.py')),
                '--quality', str(root / 'quality_comparison.json'),
                '--template', '/root/autodl-fs/dspark-verify-all-ab-20261008-h47/manifest.json',
                '--checkpoint', '/root/autodl-tmp/models/Qwen3-14B-NVFP4-20261008-h63',
                '--output', str(root / 'steady_ab')]
            with (root / 'steady_ab.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                status['steady_supervisor_pid'] = process.pid
                output.write_text(json.dumps(status, indent=2))
                code = process.wait()
            status.update(status='qualified_fixed_subsets_and_one_cell' if code == 0 else 'benefit_not_proven',
                          steady_exit_code=code)
    status['finished_at'] = time.time()
    output.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    finish(p.parse_args().root)
