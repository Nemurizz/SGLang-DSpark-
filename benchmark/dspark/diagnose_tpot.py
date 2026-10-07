"""One-cell cause diagnosis only; NEVER a gain benchmark or numerical change.

Run outside the repository using preserved helpers. Compare full, paired hot,
and paired with ordinary per-request flush, only p2048_c1 / 128 output tokens.
All requests must match frozen tokens and five acceptance fields. No retries.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import profile_command, sha, validate_row
from stable_cache_validation import start

HEAD = '0577109f4e659b8ef2ef3f979f1fd7f079b25821'


class Telemetry:
    def __init__(self, folder):
        self.folder, self.process, self.thread = folder, None, None

    def begin(self):
        query = 'timestamp,uuid,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active'
        self.stream = (self.folder / 'device_telemetry.csv').open('x')
        self.process = subprocess.Popen(['nvidia-smi', '--query-gpu=' + query,
            '--format=csv', '-lms', '200'], stdout=self.stream, stderr=subprocess.STDOUT)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.stream.close()


async def run(cli):
    import aiohttp
    template = json.loads((cli.template / 'manifest.json').read_text())
    qualified = template['qualified_manifest']
    args = SimpleNamespace(**template['args'])
    args.observe_work = False
    repo = Path(args.repo)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != HEAD:
        raise ValueError('Requires last qualified baseline; no checkout mutations')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
        raise ValueError('Tracked working tree must be clean')
    frozen = {**template['frozen'], **args.source_sha256,
        **args.execution_source_sha256, **args.preserved_untracked_source_sha256}
    def integrity():
        for path, digest in frozen.items():
            if sha(Path(path) if path.startswith('/') else repo / path) != digest:
                raise ValueError('Preserved source/reference changed: ' + path)
    integrity()
    from fixed_fa4_policy_hook import __file__ as policy_path
    if sha(policy_path) != qualified['fixed_policy_sha256']:
        raise ValueError('Qualified numerical policy changed')
    for directory, expected in qualified['model_metadata_sha256'].items():
        if {p.name: sha(p) for p in Path(directory).glob('*.json')} != expected:
            raise ValueError('Model metadata changed')
    if sha(args.sps_table) != qualified['sps_sha256']:
        raise ValueError('SPS table changed')
    if {k: importlib.metadata.version(k) for k in qualified['packages']} != qualified['packages']:
        raise ValueError('Packages changed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'], text=True).strip()
    if gpu != qualified['gpu_identity']:
        raise ValueError('Qualified GPU changed')
    dataset = json.loads((cli.template / 'dataset.json').read_text())
    spec = dataset['p2048_c1']
    refs = ab.load_frozen_references(Path(args.oracle_jsonl), dataset)
    cli.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'scope': 'p2048_c1 cause diagnosis only; NOT performance A/B',
              'nsys': cli.nsys, 'modes': cli.modes.split(',')}
    ab.dump(cli.output / 'manifest.json', {'args': vars(cli) | {'output': str(cli.output), 'template': str(cli.template)},
        'template': str(cli.template), 'head': HEAD, 'qualified_gpu': gpu, 'frozen': frozen,
        'profile_id': args.profile_id, 'observer_sha256': sha(Path(__file__).with_name('tpot_trace_hook.py')),
        'no_tensor_copies_or_new_synchronization': True, 'created_at': time.time()})
    try:
        for index, mode in enumerate(cli.modes.split(',')):
            if mode not in ('full', 'paired_hot', 'paired_flush', 'paired_toggle'):
                raise ValueError('Invalid mode')
            integrity()
            folder = cli.output / f'{index:02d}_{mode}'
            folder.mkdir()
            args.output = str(folder)
            server_mode = 'full' if mode == 'full' else 'paired'

            class DiagnosticServer(ab.ManagedServer):
                def command(self):
                    cmd, env = profile_command(*super().command(), ordinary=False,
                        profile_id=args.profile_id, runtime_folder=folder / 'runtime',
                        hook_folder=Path(__file__).resolve().parent, fixed_tile=[64, 64])
                    i = cmd.index('--forward-hooks') + 1
                    hooks = json.loads(cmd[i])
                    hooks[0]['hook_factory'] = 'tpot_trace_hook:make_hook'
                    hooks[0]['config'].update(diagnostic_folder=str(folder), nsys=cli.nsys)
                    cmd[i] = json.dumps(hooks)
                    env['SGLANG_DSPARK_TPOT_TRACE'] = '1'
                    env['PYTHONPATH'] += ':' + str(Path(ab.__file__).resolve().parent)
                    if cli.nsys:
                        cmd = [cli.nsys_bin, 'profile', '--trace=' + cli.nsys_trace + ',nvtx', '--sample=none', '--cpuctxsw=none',
                            '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--kill=none',
                            '--wait=all', '--flush-on-cudaprofilerstop=true', '--cuda-event-trace=false',
                            '--output=' + str(folder / 'trace')] + cmd
                    return cmd, env

            server = DiagnosticServer(args, server_mode, folder)
            telemetry = Telemetry(folder)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                        connector=ab.http_connector(aiohttp)) as session:
                    telemetry.begin()
                    base = await start(server, session)

                    async def checked(phase, ordinal):
                        effective = mode
                        if mode == 'paired_toggle':
                            effective = ('paired_flush' if phase == 'prime' or
                                (phase.startswith('diagnostic_') and ordinal % 4 in (0, 3)) else 'paired_hot')
                        if effective == 'paired_flush':
                            await ab.api(session, base, '/flush_cache?timeout=10', 'POST')
                        row = await ab.generate(session, base, spec['prompts'][0], args.output_tokens, args.seed)
                        row.update(cell='p2048_c1', variant=0, phase=phase, mode=effective, ordinal=ordinal,
                                   recorded_wall_time=time.time())
                        ab.append_jsonl(folder / 'requests.jsonl', row)
                        validate_row(row, spec, args.output_tokens, ordinary=False,
                                     require_cold=effective != 'paired_hot' or phase == 'prime')
                        ab.check_reference(refs[('p2048_c1', 0)], row)
                        if effective == 'paired_hot' and phase != 'prime' and row['cached_tokens'] != 2047:
                            raise ab.SafetyFailure('Requires exact paired hot hit')
                        server.check()
                        return row

                    await checked('prime', 0)
                    for n in range(cli.settle):
                        await checked('settle_excluded', n)
                    ab.dump(folder / 'trace_start', {'time': time.time()})
                    rows = []
                    for n in range(cli.requests):
                        rows.append(await checked('diagnostic_profiled' if cli.nsys else 'diagnostic_host_observed', n))
                    ab.dump(folder / 'trace_stop', {'time': time.time()})
                    await checked('drain_excluded', 0)  # Flush buffered host records before excluded prefill.
                    if not (folder / 'worker_trace.jsonl').exists():
                        raise ab.SafetyFailure('No completed worker trace')
                    summary = {metric: ab.distribution([r[metric] for r in rows])
                               for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
                    ab.dump(folder / 'summary.json', summary)
                    print('COMPLETE ' + folder.name + ' ' + json.dumps(summary), flush=True)
            finally:
                server.stop()
                telemetry.stop()
                # nsys can finish exporting before HTTP children finish draining.
                # Never start an arm while an old owned listener is still open.
                for attempt in range(20):
                    with socket.socket() as sock:
                        open_port = sock.connect_ex(('127.0.0.1', args.port)) == 0
                    if not open_port:
                        break
                    await asyncio.sleep(.5)
                else:
                    raise ab.SafetyFailure('Owned diagnostic listener did not finish draining')
            integrity()
            status.setdefault('completed', []).append(folder.name)
            ab.dump(cli.output / 'status.json', status)
        status['status'] = 'diagnosis_capture_complete'
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        integrity()
        status.update(frozen_sources_and_references_unchanged=True, finished_at=time.time())
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'diagnosis_capture_complete' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', default='full,paired_hot,paired_flush')
    parser.add_argument('--requests', type=int, default=6)
    parser.add_argument('--settle', type=int, default=4)
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--nsys-bin', default='nsys', help='Explicit parallel-install CLI; no global PATH change')
    parser.add_argument('--nsys-trace', choices=['cuda', 'cuda-sw'], default='cuda')
    cli = parser.parse_args()
    if not 1 <= cli.requests <= 32 or cli.settle < 2:
        parser.error('Bounded cause diagnosis: 1..32 requests, >=2 settle')
    raise SystemExit(asyncio.run(run(cli)))
