"""Real quality prerequisite then separate one-cell unprofiled BF16/NVFP4 ABBA."""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import time

import dspark_prefix_ab as ab
from quality_data import sha
from quality_run import QualityServer, validate
from quant_feasibility_gate import baseline_identity, launch_args_from_template


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


def admitted(groups):
    if len(groups) < 3:
        return False
    means = [group['mean'] for group in groups[-3:]]
    return (max(means)/min(means)-1 <= .005 and
            all(group['p95']/group['p50'] <= 1.02 for group in groups[-3:]))


async def drain_listener(port):
    for _ in range(20):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return
        await asyncio.sleep(.5)
    raise ab.SafetyFailure('Owned listener did not finish draining')

PROTOCOL = {'cell': 'p2048_c1', 'output_tokens': 128, 'concurrency': 1,
    'equal_kv_capacity_max_total_tokens': 16384,
    'warmup_min': 24, 'warmup_max': 64, 'warmup_group': 8,
    'warmup_mean_spread_limit': .005, 'warmup_p95_over_p50_limit': 1.02,
    'control_mean_spread_limit': .005, 'control_p95_spread_limit': .01,
    'order': ['bf16', 'candidate', 'candidate', 'bf16'], 'steady_requests_per_window': 64,
    'minimum_e2e_gain_in_both_orders': .01, 'p95_regression_tolerance': 0,
    'no_retries_or_discarded_steady_observations': True,
    'graphs_cache_adaptive_draft_kv_quantization': 'OFF',
    'performance_scope': 'single p2048/c1/128 synthetic fixed workload; not fleet/high-concurrency proof'}


def noise(left, right):
    result = {}
    for metric in ('tpot_ms', 'e2e_ms'):
        result[metric] = {stat: max(left[metric][stat], right[metric][stat]) /
                         min(left[metric][stat], right[metric][stat]) - 1 for stat in ('mean', 'p95')}
    result['admitted'] = all(item['mean'] <= .005 and item['p95'] <= .01
                             for key, item in result.items() if key != 'admitted')
    return result


class EqualCapacityServer(QualityServer):
    def command(self):
        cmd, env = super().command()
        return cmd + ['--max-total-tokens', str(PROTOCOL['equal_kv_capacity_max_total_tokens'])], env


async def arm(args, folder, prompt, windows, oracle, ordinary=False):
    import aiohttp
    folder.mkdir()
    server = EqualCapacityServer(args, 'full', folder)
    telemetry = DeviceTelemetry(folder)
    sampler = ab.GPUSampler(folder / 'gpu.jsonl', 200)
    summaries = []
    try:
        telemetry.begin()
        sampler.start()
        server.start()
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp),
                timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as session:
            base = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, '/model_info')
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() >= deadline:
                        raise ab.SafetyFailure('Startup timeout')
                    await asyncio.sleep(.5)
            ab.dump(folder / 'server_info.json', await ab.api(session, base, '/server_info'))

            async def checked(phase, index, reference=None):
                task = asyncio.create_task(ab.generate(session, base, prompt, 128, args.seed))
                try:
                    while not task.done():
                        await asyncio.wait([task], timeout=.5)
                        server.check()
                    row = await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                row.update(phase=phase, index=index, cell='p2048_c1', variant=0, concurrency=1,
                           quant=args.quant, ordinary=args.ordinary)
                ab.append_jsonl(folder / 'requests.jsonl', row)
                if row['cached_tokens'] or row['meta_info'].get('num_retractions', 0):
                    raise ab.SafetyFailure('Cache/retraction in timing workload')
                if reference:
                    ab.check_reference(reference, row, check_acceptance=not ordinary and not reference.get('ordinary'))
                server.check()
                return row

            first = await checked('prime_excluded', 0, oracle)
            validate(folder, args.quant, args.ordinary, args.fp4_backend, args.recipe)
            info = json.loads((folder / 'server_info.json').read_text())
            if info.get('max_total_tokens') != PROTOCOL['equal_kv_capacity_max_total_tokens']:
                raise ab.SafetyFailure('Equal KV-capacity launch flag was not resolved')
            if ordinary:
                await checked('reference_repeat', 1, first)
                return first
            if oracle.get('ordinary'):
                oracle = first
            groups = []
            for group in range(8):
                rows = [await checked('warmup_excluded', group*8 + i, oracle) for i in range(8)]
                item = ab.distribution([r['tpot_ms'] for r in rows])
                groups.append(item)
                ab.append_jsonl(folder / 'warmup_groups.jsonl', item)
                print('WARMUP ' + folder.name + ' ' + json.dumps(item), flush=True)
                if admitted(groups):
                    break
            else:
                raise ab.SafetyFailure('Warmup failed fixed admission; do not retry/select')
            ab.dump(folder / 'warmup_admission.json', {'admitted': True, 'groups': groups})
            for window in range(windows):
                sample_start = sampler.begin('steady-' + str(window))
                before = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                began = time.perf_counter()
                rows = []
                for i in range(64):
                    rows.append(await checked('steady-' + str(window), i, oracle))
                    if (i+1) % 8 == 0:
                        print('STEADY ' + folder.name + f' w{window} n{i+1}', flush=True)
                after = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                item = ab.aggregate('full', 'p2048_c1', window, rows, before, after,
                    sampler.summary(sample_start), time.perf_counter() - began)
                item.update(recipe=args.recipe if args.quant else 'bf16', timing_profiler=False)
                summaries.append(item)
                ab.append_jsonl(folder / 'summaries.jsonl', item)
                print('WINDOW ' + folder.name + ' ' + json.dumps(item), flush=True)
    finally:
        server.stop()
        sampler.stop()
        telemetry.stop()
        await drain_listener(args.port)
    return summaries, oracle


async def run(cli):
    quality = json.loads(cli.quality.read_text())
    if quality['quality_gate'] != 'passed_for_fixed_subsets_only':
        raise ValueError('Cannot run benefit proof after observed quality loss')
    if cli.output.exists():
        raise ValueError('Fresh timing folder required; no retry')
    args = launch_args_from_template(json.loads(cli.template.read_text()))
    args.tokenizer = args.model
    args.ordinary = False
    args.quant = False
    args.fp4_backend = cli.fp4_backend
    args.recipe = cli.recipe
    if quality.get('candidate_recipe', 'nvfp4') != args.recipe:
        raise ValueError('Quality evidence belongs to another quantization recipe')
    expected_backend = 'sgl_kernel_cutlass_fp8' if args.recipe == 'w8a8_fp8_native' else args.fp4_backend
    if quality.get('candidate_backend', 'flashinfer_cutlass') != expected_backend:
        raise ValueError('Quality evidence belongs to another backend')
    args.startup_timeout, args.request_timeout = 900, 45
    args.enable_standard_graphs = False
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    old_model = args.model
    candidate_model = str(cli.checkpoint / 'model') if args.recipe == 'nvfp4' else old_model
    before = baseline_identity(args.repo)
    old_dataset = Path(args.oracle_jsonl).parent / 'dataset.json'
    prompt = json.loads(old_dataset.read_text())['p2048_c1']['prompts'][0]
    old_refs = [json.loads(line) for line in Path(args.oracle_jsonl).read_text().splitlines()]
    baseline_oracle = next(r for r in old_refs if r['cell']=='p2048_c1' and r['variant']==0)
    cli.output.mkdir(parents=True)
    frozen = {str(path): sha(path) for path in (cli.quality, cli.template, old_dataset, Path(args.oracle_jsonl))}
    ab.dump(cli.output / 'manifest.json', {'protocol': PROTOCOL, 'args': vars(args),
        'frozen_sha256': frozen, 'baseline_before': before,
        'quality_scope': quality['scope'], 'quality_gate': quality['quality_gate'],
        'tools_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'created_at': time.time(), 'performance_claim': 'unproven'})
    status = {'status': 'running', 'performance_claim': 'unproven'}
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip():
            raise ab.SafetyFailure('GPU occupied')
        args.quant, args.ordinary = True, True
        args.model = candidate_model
        quant_ar = await arm(args, cli.output / 'quant_ar_oracle', prompt, 0, None, True)
        ab.dump(cli.output / 'quant_ar_oracle.json', quant_ar)
        args.quant, args.ordinary, args.model = False, False, old_model
        control, _ = await arm(args, cli.output / '00_bf16_control', prompt, 2, baseline_oracle)
        check = noise(*control)
        ab.dump(cli.output / 'control_noise.json', check)
        if not check['admitted']:
            raise ab.SafetyFailure('BF16/BF16 temporal noise failed; candidate not timed')
        status['control_admitted'] = True
        results, quant_oracle = {}, quant_ar
        for ordinal, recipe in enumerate(PROTOCOL['order'], 1):
            args.quant = recipe == 'candidate'
            args.model = candidate_model if args.quant else old_model
            summaries, oracle = await arm(args, cli.output / f'{ordinal:02d}_{recipe}', prompt, 1,
                quant_oracle if args.quant else baseline_oracle)
            if args.quant:
                quant_oracle = oracle
            results[ordinal] = summaries[0]
            ab.append_jsonl(cli.output / 'summaries.jsonl', {'arm': ordinal, **summaries[0]})
            if ordinal in (2, 4):
                baseline, candidate = ((results[1], results[2]) if ordinal==2 else (results[4], results[3]))
                ab.check_p95(baseline, candidate, 64, 0)
                ratios = {metric: {stat: candidate[metric][stat]/baseline[metric][stat]
                                   for stat in ('mean','p95')} for metric in ('ttft_ms','tpot_ms','e2e_ms')}
                comparison = {'order': 'AB' if ordinal==2 else 'BA', 'ratios': ratios,
                              'p95_guard': 'passed', 'e2e_gain': 1-ratios['e2e_ms']['mean']}
                ab.append_jsonl(cli.output / 'comparisons.jsonl', comparison)
                if comparison['e2e_gain'] <= .01:
                    raise ab.SafetyFailure('This order did not improve E2E by >1%; stop candidate')
        drift = noise(results[1], results[4])
        ab.dump(cli.output / 'control_drift.json', drift)
        if not drift['admitted']:
            raise ab.SafetyFailure('BF16 AB/BA drift failed; no benefit claim')
        status.update(status='passed_quality_and_one_cell_counterbalanced_benefit',
            performance_claim='>1% E2E BOTH orders, no p95 regression; specified cell and fixed quality subsets only')
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        status.update(baseline_after=baseline_identity(args.repo), finished_at=time.time(),
                      frozen_inputs_unchanged=all(sha(path)==value for path,value in frozen.items()))
        if status['baseline_after'] != before or not status['frozen_inputs_unchanged']:
            status.update(status='safety_stopped', error='Immutable input/source changed')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'].startswith('passed_') else 2


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fp4-backend', default='flashinfer_cutlass', choices=['flashinfer_cutlass', 'flashinfer_cudnn'])
    p.add_argument('--recipe', default='nvfp4', choices=['nvfp4', 'w8a8_fp8_native'])
    raise SystemExit(asyncio.run(run(p.parse_args())))
