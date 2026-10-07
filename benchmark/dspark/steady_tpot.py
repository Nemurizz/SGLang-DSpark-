"""Prespecified one-cell unprofiled control and AB/BA; no numerical edits.

Requires unchanged qualified sources/packages/oracles and baseline HEAD. Timing
uses only the already-qualified fixed FA4 policy/attestation, NOT diagnostic
NVTX/work hooks. Never discard steady observations or relax candidate p95.
"""
import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import sha, validate_row
from stable_cache_validation import QualifiedServer, start

HEAD = '0577109f4e659b8ef2ef3f979f1fd7f079b25821'
PROTOCOL = {
    'cell': 'p2048_c1', 'output_tokens': 128,
    'timing_has_profiler_or_diagnostic_hooks': False,
    'warmup_min': 24, 'warmup_max': 64, 'warmup_group': 8,
    'warmup_last_groups': 3, 'warmup_mean_spread_limit': .005,
    'warmup_p95_over_p50_limit': 1.02,
    'steady_requests_per_window': 64,
    'control_mean_spread_limit': .005, 'control_p95_spread_limit': .01,
    'control_limits_are_noise_admission_NOT_candidate_tolerance': True,
    'candidate_p95_tolerance': 0.0,
    'candidate_order': ['full', 'paired', 'paired', 'full'],
    'all_cold_warmup_and_steady_records_preserved': True,
    'no_request_retries': True, 'performance_claim': 'unproven',
}


def admitted(groups):
    if len(groups) < 3:
        return False
    means = [group['mean'] for group in groups[-3:]]
    return ((max(means) / min(means) - 1) <= PROTOCOL['warmup_mean_spread_limit']
        and all(group['p95'] / group['p50'] <= PROTOCOL['warmup_p95_over_p50_limit']
                for group in groups[-3:]))


def control_check(left, right):
    means = [left['tpot_ms']['mean'], right['tpot_ms']['mean']]
    tails = [left['tpot_ms']['p95'], right['tpot_ms']['p95']]
    result = {'mean_spread': max(means) / min(means) - 1,
              'p95_spread': max(tails) / min(tails) - 1}
    result['admitted'] = (result['mean_spread'] <= PROTOCOL['control_mean_spread_limit']
        and result['p95_spread'] <= PROTOCOL['control_p95_spread_limit'])
    return result


class DeviceTelemetry:
    def __init__(self, folder):
        self.folder, self.process, self.stream = folder, None, None

    def begin(self):
        query = ('timestamp,uuid,memory.used,pstate,clocks.sm,clocks.mem,'
                 'power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active')
        self.stream = (self.folder / 'device_telemetry.csv').open('x')
        self.process = subprocess.Popen(['nvidia-smi', '--query-gpu=' + query,
            '--format=csv', '-lms', '200'], stdout=self.stream, stderr=subprocess.STDOUT)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.stream.close()


async def drain_listener(port):
    for _ in range(20):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return
        await asyncio.sleep(.5)
    raise ab.SafetyFailure('Owned listener did not finish draining')


def integrity(template, args):
    repo = Path(args.repo)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != HEAD:
        raise ab.SafetyFailure('Requires last qualified baseline; no checkout mutation')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=repo, text=True).strip():
        raise ab.SafetyFailure('Tracked working tree changed')
    frozen = {**template['frozen'], **args.source_sha256,
              **args.execution_source_sha256, **args.preserved_untracked_source_sha256}
    for path, digest in frozen.items():
        if sha(Path(path) if path.startswith('/') else repo / path) != digest:
            raise ab.SafetyFailure('Qualified source/reference changed: ' + path)
    qualified = template['qualified_manifest']
    from fixed_fa4_policy_hook import __file__ as policy_path
    if sha(policy_path) != qualified['fixed_policy_sha256']:
        raise ab.SafetyFailure('Qualified numerical policy changed')
    for directory, expected in qualified['model_metadata_sha256'].items():
        if {p.name: sha(p) for p in Path(directory).glob('*.json')} != expected:
            raise ab.SafetyFailure('Qualified model metadata changed')
    if sha(args.sps_table) != qualified['sps_sha256']:
        raise ab.SafetyFailure('Qualified SPS table changed')
    if {k: importlib.metadata.version(k) for k in qualified['packages']} != qualified['packages']:
        raise ab.SafetyFailure('Qualified packages changed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'], text=True).strip()
    if gpu != qualified['gpu_identity']:
        raise ab.SafetyFailure('Qualified GPU/driver changed')


async def arm(args, template, refs, spec, root, ordinal, mode, windows):
    import aiohttp
    import os
    integrity(template, args)
    folder = root / f'{ordinal:02d}_{mode}'
    folder.mkdir()
    args.output = str(folder)
    server = QualifiedServer(args, mode, folder)
    telemetry = DeviceTelemetry(folder)
    summaries = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                connector=ab.http_connector(aiohttp)) as session:
            telemetry.begin()
            base = await start(server, session)

            async def checked(phase, number, window=None):
                began = time.time()
                row = await ab.generate(session, base, spec['prompts'][0], args.output_tokens, args.seed)
                row.update(cell='p2048_c1', variant=0, concurrency=1, phase=phase, ordinal=number,
                    window=window, mode=mode, arm=ordinal, began_wall_time=began,
                    finished_wall_time=time.time(), cpu_load_1_5_15=list(os.getloadavg()))
                ab.append_jsonl(folder / 'requests.jsonl', row)
                validate_row(row, spec, args.output_tokens, ordinary=False,
                             require_cold=mode == 'full' or phase == 'prime_excluded')
                ab.check_reference(refs[('p2048_c1', 0)], row)
                if mode == 'paired' and phase != 'prime_excluded' and row['cached_tokens'] != 2047:
                    raise ab.SafetyFailure('Timing requires exact paired hot prefix')
                if row['meta_info'].get('num_retractions') != 0:
                    raise ab.SafetyFailure('Timing requires zero retractions')
                server.check()
                return row

            await checked('prime_excluded', 0)
            groups = []
            for group in range(PROTOCOL['warmup_max'] // PROTOCOL['warmup_group']):
                rows = [await checked('warmup_excluded', group * 8 + i) for i in range(8)]
                item = ab.distribution([row['tpot_ms'] for row in rows])
                groups.append(item)
                ab.append_jsonl(folder / 'warmup_groups.jsonl', {'group': group, **item})
                print('WARMUP ' + folder.name + ' ' + json.dumps(item), flush=True)
                if admitted(groups):
                    break
            else:
                raise ab.SafetyFailure('Prespecified warmup stability not reached; no steady selection')
            ab.dump(folder / 'warmup_admission.json', {'groups': groups, 'admitted': True,
                                                       'protocol': PROTOCOL})
            for window in range(windows):
                before = ab.info_stats(await ab.api(session, base, '/server_info'), required=mode == 'paired')
                began = time.perf_counter()
                rows = []
                for i in range(64):
                    rows.append(await checked('steady', i, window))
                    if (i + 1) % 8 == 0:
                        ab.append_jsonl(folder / 'steady_groups.jsonl', {'window': window, 'group': i // 8,
                            'tpot_ms': ab.distribution([row['tpot_ms'] for row in rows[-8:]])})
                        print('STEADY ' + folder.name + f' window={window} n={i+1}', flush=True)
                after = ab.info_stats(await ab.api(session, base, '/server_info'), required=mode == 'paired')
                item = ab.aggregate(mode, 'p2048_c1', window, rows, before, after,
                    {'scope': 'device_telemetry.csv; device-wide, not allocator peak'},
                    time.perf_counter() - began)
                item.update(arm=ordinal, window=window, timing_profiler=False, diagnostic_hooks=False)
                ab.append_jsonl(root / 'summaries.jsonl', item)
                summaries.append(item)
                print('WINDOW ' + folder.name + ' ' + json.dumps({k: item[k] for k in
                    ('request_count', 'ttft_ms', 'tpot_ms', 'e2e_ms')}), flush=True)
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
    integrity(template, args)
    return summaries


async def run(cli):
    template = json.loads((cli.template / 'manifest.json').read_text())
    args = SimpleNamespace(**template['args'])
    args.observe_work = False
    if args.output_tokens != 128:
        raise ValueError('Only qualified 128-output profile supported')
    integrity(template, args)
    dataset = json.loads((cli.template / 'dataset.json').read_text())
    refs = ab.load_frozen_references(Path(args.oracle_jsonl), dataset)
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / 'manifest.json', {'template': str(cli.template), 'protocol': PROTOCOL,
        'template_sha256': sha(cli.template / 'manifest.json'), 'tool_sha256': sha(__file__),
        'head': HEAD, 'profile_id': args.profile_id, 'created_at': time.time()})
    status = {'status': 'running', 'performance_claim': 'unproven'}
    try:
        full = await arm(args, template, refs, dataset['p2048_c1'], cli.output, 0, 'full', 2)
        control = control_check(*full)
        ab.dump(cli.output / 'control_noise.json', control)
        if not control['admitted']:
            raise ab.SafetyFailure('Full/full temporal noise exceeds admission; candidate not started')
        status['full_full_noise_admitted'] = True
        if cli.control_only:
            status['status'] = 'control_complete_only'
        else:
            result = {}
            for ordinal, mode in enumerate(PROTOCOL['candidate_order'], 1):
                result[ordinal] = (await arm(args, template, refs, dataset['p2048_c1'],
                                             cli.output, ordinal, mode, 1))[0]
                if ordinal in (2, 4):
                    control, candidate = ((result[1], result[2]) if ordinal == 2
                                          else (result[4], result[3]))
                    ab.check_p95(control, candidate, 64, 0.0)
                    ab.append_jsonl(cli.output / 'comparisons.jsonl', {'order': 'AB' if ordinal == 2 else 'BA',
                        'p95_guard': 'passed', 'ratios_paired_over_full': {
                            metric: {stat: candidate[metric][stat] / control[metric][stat]
                                     for stat in ('mean', 'p95')}
                            for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}})
            status['status'] = 'one_cell_counterbalanced_complete_NOT_general_gain_proof'
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        integrity(template, args)
        status.update(qualified_inputs_unchanged=True, finished_at=time.time())
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 2 if status['status'] == 'safety_stopped' else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-only', action='store_true')
    raise SystemExit(asyncio.run(run(parser.parse_args())))
