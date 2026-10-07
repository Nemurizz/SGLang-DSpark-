"""Audit saved one-cell recheck, without dropping slow observations."""
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import dspark_prefix_ab as ab
from stable_baseline_gate import sha

ROOT = Path('/root/autodl-fs')
INSTALL = ROOT / 'dspark-nsys-2025.6.3.541-20261008-h40'
STEADY = ROOT / 'dspark-steady-tpot-20261008-h41'
TRACE = ROOT / 'dspark-nsys-full-baseline-20261008-h42'
HARDWARE = ROOT / 'dspark-nsys-full-hardware-20261008-h43'
TEMPLATE = Path('/root/autodl-tmp/results/dspark-cache-balanced-ab-20261007-h34')


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def telemetry_summary(folder, request_rows):
    samples = list(csv.DictReader((folder / 'device_telemetry.csv').open(), skipinitialspace=True))
    ranges = [(r.get('began_wall_time', r['meta_info']['response_sent_to_client_ts'] - r['e2e_ms']/1000),
               r.get('finished_wall_time', r['meta_info']['response_sent_to_client_ts'])) for r in request_rows]
    selected = [s for s in samples if any(a <= datetime.strptime(s['timestamp'],
        '%Y/%m/%d %H:%M:%S.%f').timestamp() <= b for a,b in ranges)]
    values = {'sample_count': len(selected), 'sample_interval_ms': 200,
              'scope': 'device-wide sampled observations; does not exclude sub-200ms transients'}
    for name in ('pstate', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]', 'clocks_event_reasons.active'):
        values[name] = dict(Counter(s[name] for s in selected))
    if selected and 'memory.used [MiB]' in selected[0]:
        values['peak_device_memory_mib'] = max(int(s['memory.used [MiB]'].split()[0]) for s in selected)
    values['temperature_c'] = ab.distribution([int(s['temperature.gpu'].split()[0]) for s in selected])
    values['power_watts'] = ab.distribution([float(s['power.draw [W]'].split()[0]) for s in selected])
    return values


template = load(TEMPLATE / 'manifest.json')
dataset = load(TEMPLATE / 'dataset.json')
refs = ab.load_frozen_references(Path(template['args']['oracle_jsonl']), dataset)
result = {'scope': '2K c1 only; full/full noise control + full-only collector diagnosis',
          'performance_claim': 'unproven', 'original_h34_failure_not_revoked': True,
          'installation': {'package_sha256': sha(INSTALL / 'nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb'),
              'new_binary': str(INSTALL / 'package/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys'),
              'global_binary_resolved': subprocess.check_output(['readlink','-f','/usr/local/bin/nsys'],text=True).strip(),
              'old_version': subprocess.check_output(['nsys','--version'],text=True).strip(),
              'smoke_hw': load(INSTALL / 'smoke_hw_audit.json'),
              'smoke_sw': load(INSTALL / 'smoke_sw_audit.json')},
          'steady_status': load(STEADY / 'status.json'), 'noise_control': load(STEADY / 'control_noise.json'),
          'prespecified_protocol': load(STEADY / 'manifest.json')['protocol'],
          'steady_summaries': rows(STEADY / 'summaries.jsonl'),
          'profile_status': load(TRACE / 'status.json'),
          'profile_analysis': load(TRACE / '00_full/analysis.json'),
          'profile_collector_audit': load(TRACE / '00_full/collector_audit.json'),
          'hardware_profile_status': load(HARDWARE / 'status.json'),
          'hardware_profile_analysis': load(HARDWARE / '00_full/analysis.json'),
          'hardware_collector_audit': load(HARDWARE / '00_full/collector_audit.json')}
assert result['profile_analysis']['per_step_launch_geometry'] == result['hardware_profile_analysis']['per_step_launch_geometry']
result['software_hardware_decode_geometry_identical'] = True
checked = 0
for name, folder in [('unprofiled_full', STEADY / '00_full'), ('profiled_full', TRACE / '00_full'),
                     ('hardware_profiled_full', HARDWARE / '00_full')]:
    requests = rows(folder / 'requests.jsonl')
    for row in requests:
        ab.check_reference(refs[('p2048_c1', 0)], row)
        assert row['cached_tokens'] == 0 and row['meta_info']['num_retractions'] == 0
        assert row['prompt_tokens'] == 2048 and len(row['output_ids']) == 128
    checked += len(requests)
    result.setdefault('request_audit', {})[name] = {'checked_requests': len(requests),
        'phases': dict(Counter(r['phase'] for r in requests)),
        'all_128_tokens_and_five_acceptance_fields_match': True}
    if name == 'unprofiled_full':
        steady = [r for r in requests if r['phase'] == 'steady']
        assert len(steady) == 128
        result['unprofiled_telemetry'] = telemetry_summary(folder, steady)
        result['all_steady_stream_patterns_identical'] = len({tuple(e['cumulative_tokens'] for e in
            r['stream_events']) for r in steady}) == 1
        result['windows'] = {}
        for window in (0, 1):
            selected = [r for r in steady if r['window'] == window]
            result['windows'][str(window)] = {
                'client_minus_server_e2e_ms': ab.distribution([r['e2e_ms'] - r['meta_info']['e2e_latency'] * 1000 for r in selected]),
                'server_e2e_ms': ab.distribution([r['meta_info']['e2e_latency'] * 1000 for r in selected]),
                'cpu_load_1': ab.distribution([r['cpu_load_1_5_15'][0] for r in selected]),
                'telemetry': telemetry_summary(folder, selected)}
        # A descriptive threshold only, NEVER excludes requests from statistics.
        result['slow_request_examples_scope'] = 'descriptive TPOT > 19.3ms, not a selection or admission rule'
        result['slow_request_examples'] = [{'window': r['window'], 'ordinal': r['ordinal'],
            'tpot_ms': r['tpot_ms'], 'server_e2e_ms': r['meta_info']['e2e_latency'] * 1000,
            'client_minus_server_e2e_ms': r['e2e_ms'] - r['meta_info']['e2e_latency'] * 1000,
            'cpu_load_1': r['cpu_load_1_5_15'][0], 'telemetry': telemetry_summary(folder,[r])}
            for r in steady if r['tpot_ms'] > 19.3]
result['checked_requests_total'] = checked
repo = Path(template['args']['repo'])
result['final_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
result['final_main'] = subprocess.check_output(['git','rev-parse','main'],cwd=repo,text=True).strip()
result['tracked_status'] = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
result['untracked_count'] = len(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo).splitlines())
frozen = {**template['frozen'], **template['args']['source_sha256'],
          **template['args']['execution_source_sha256'], **template['args']['preserved_untracked_source_sha256']}
result['frozen_inputs_unchanged'] = all(sha(Path(p) if p.startswith('/') else repo / p) == d for p,d in frozen.items())
result['frozen_input_count'] = len(frozen)
result['oracle_sha256'] = sha(template['args']['oracle_jsonl'])
result['baseline_rollback_retained'] = result['final_head'] == '0577109f4e659b8ef2ef3f979f1fd7f079b25821'
assert result['frozen_inputs_unchanged'] and result['baseline_rollback_retained'] and not result['tracked_status']
path = ROOT / 'dspark-nsys-recheck-summary-20261008-h40-h43.json'
with path.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({k: v for k,v in result.items() if k not in
    ('profile_analysis','profile_collector_audit','hardware_profile_analysis','hardware_collector_audit',
     'installation','prespecified_protocol','slow_request_examples')}, indent=2))
