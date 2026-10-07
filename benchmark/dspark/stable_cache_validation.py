"""Qualified-profile recovery pressure gate and counterbalanced hot-cache A/B.

No numerical changes, no graph/kernel/quantization experiments. Never overwrite
references. Correctness failures stop the owned server. Performance runs exclude
priming, use fixed request waves, fresh connections, no retries, and AB/BA order.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import profile_command, sha, validate_profile, validate_row
from stable_profile_hook import validate_runtime


def balanced_order(pairs):
    if pairs < 2 or pairs % 2:
        raise ValueError('Use an even number of paired blocks, at least two')
    return [('full', 'paired') if i % 2 == 0 else ('paired', 'full') for i in range(pairs)]


def pressure_prompts(dataset):
    base = dataset['p8192_c8']['prompts'][0]
    tokens = sorted(set(base) - {base[0]})[:6]
    if len(tokens) != 6:
        raise ValueError('Cannot build six distinct valid-token pressure prefixes')
    return [[token] + base[1:] for token in tokens]


def bootstrap_ratio(left, right, seed, iterations=2000):
    """Paired-wave cluster bootstrap; not an IID request confidence interval."""
    if len(left) != len(right) or len(left) < 8:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(iterations):
        indices = [rng.randrange(len(left)) for _ in left]
        denominator = sum(left[i] for i in indices)
        ratios.append(sum(right[i] for i in indices) / denominator)
    return {'ratio': sum(right) / sum(left),
            'ci95_low': ab.percentile(ratios, .025), 'ci95_high': ab.percentile(ratios, .975),
            'resampling_unit': 'matched concurrency wave', 'iterations': iterations}


class QualifiedServer(ab.ManagedServer):
    def command(self):
        cmd, env = super().command()
        cmd, env = profile_command(cmd, env, ordinary=False, profile_id=self.args.profile_id,
            runtime_folder=self.folder / 'runtime', hook_folder=Path(__file__).resolve().parent,
            fixed_tile=[64, 64])
        if self.args.observe_work:
            index = cmd.index('--forward-hooks') + 1
            hooks = json.loads(cmd[index])
            hooks[0]['hook_factory'] = 'recovery_work_hook:make_hook'
            hooks[0]['config']['work_folder'] = str(self.folder)
            cmd[index] = json.dumps(hooks)
            env['SGLANG_DSPARK_RECOVERY_WORK_OBSERVER'] = '1'
        return cmd, env


async def start(server, session):
    base = f'http://127.0.0.1:{server.args.port}'
    # Do not evict another job or silently benchmark alongside an external job.
    for attempt in range(10):
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        if not pids:
            break
        if attempt == 9:
            raise ab.SafetyFailure('GPU is occupied; refusing to replace external GPU work: ' + pids)
        await asyncio.sleep(1)
    server.start()
    deadline = time.monotonic() + server.args.startup_timeout
    while True:
        server.check()
        try:
            await ab.api(session, base, '/model_info')
            break
        except (asyncio.TimeoutError, __import__('aiohttp').ClientError):
            if time.monotonic() > deadline:
                raise ab.SafetyFailure('Server startup timeout')
            await asyncio.sleep(.5)
    info = await ab.api(session, base, '/server_info')
    validate_profile(info, False)
    ab.dump(server.folder / 'server_info.json', info)
    for _ in range(server.args.warmup_requests):
        await ab.generate(session, base, server.args.warmup_prompt, server.args.output_tokens, server.args.seed)
        server.check()
    records = list((server.folder / 'runtime').glob('*.json'))
    if not records:
        raise ab.SafetyFailure('Missing runtime attestation')
    for path in records:
        validate_runtime(json.loads(path.read_text()), fixed_tile=[64, 64])
    await ab.api(session, base, '/flush_cache?timeout=10', 'POST')
    print('READY ' + server.folder.name, flush=True)
    return base


async def request(session, base, args, server, spec, variant, phase, refs, **labels):
    row = await ab.generate(session, base, spec['prompts'][variant], args.output_tokens, args.seed)
    row.update(cell=labels.pop('cell'), variant=variant, phase=phase, concurrency=spec['concurrency'],
               profile_id=args.profile_id, mode=server.mode, **labels)
    ab.append_jsonl(server.folder / 'requests.jsonl', row)
    validate_row(row, spec, args.output_tokens, ordinary=False,
                 require_cold=server.mode == 'full' or phase == 'cold')
    if refs is not None:
        ab.check_reference(refs[(row['cell'], variant)], row)
    server.check()
    return row


async def wave(tasks):
    pending = [asyncio.create_task(task) for task in tasks]
    try:
        return await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def check_sources(args):
    sources = {**args.source_sha256, **getattr(args, 'execution_source_sha256', {}),
               **getattr(args, 'preserved_untracked_source_sha256', {})}
    if any(sha(Path(args.repo) / name) != digest for name, digest in sources.items()):
        raise ab.SafetyFailure('Qualified inference sources changed')


async def recovery(args, dataset, refs, root):
    import aiohttp
    fillers = pressure_prompts(dataset)
    pressure_spec = {'prompt_length': 8192, 'concurrency': 1, 'prompts': fillers}
    # Independent full-prefill DSpark controls for synthetic pressure prompts.
    pressure_refs = {}
    for mode in ('full', 'paired'):
        check_sources(args)
        ab.dump(root / 'progress.json', {'stage': 'recovery', 'mode': mode, 'phase': 'startup'})
        folder = root / ('recovery_' + mode)
        folder.mkdir()
        server = QualifiedServer(args, mode, folder)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                             connector=ab.http_connector(aiohttp)) as session:
                base = await start(server, session)
                if mode == 'full':
                    for i in range(len(fillers)):
                        row = await request(session, base, args, server, pressure_spec, i, 'pressure_reference',
                                            None, cell='pressure')
                        pressure_refs[('pressure', i)] = row
                    print('PASS full-prefill pressure controls', flush=True)
                else:
                    spec = dataset['p8192_c8']
                    for i in range(8):
                        await request(session, base, args, server, spec, i, 'prime', refs, cell='p8192_c8')
                    print('PASS paired 8K suffix priming', flush=True)
                    before = ab.info_stats(await ab.api(session, base, '/server_info'))
                    budget = before['max_tokens']
                    unique_pressure_tokens = (len(fillers) + 1) * 8192
                    conservative_peak_target_tokens = (len(fillers) + 1) * (8192 + args.output_tokens) + 8192
                    if budget >= unique_pressure_tokens or 4 * budget <= conservative_peak_target_tokens:
                        raise ab.SafetyFailure('Pressure geometry does not isolate sidecar eviction from target capacity')
                    # Keep common root recent using variant 7, leave other leaves
                    # stale. Six disjoint ~8K prefixes exceed the DEFAULT 25%
                    # sidecar budget while total target radix still fits capacity.
                    for i in range(len(fillers)):
                        ab.dump(root / 'progress.json', {'stage': 'recovery', 'mode': mode, 'phase': 'pressure', 'prefix': i})
                        await request(session, base, args, server, spec, 7, 'root_keepalive', refs, cell='p8192_c8')
                        await request(session, base, args, server, pressure_spec, i, 'pressure_fill',
                                      pressure_refs, cell='pressure')
                    for i in range(8):
                        await request(session, base, args, server, spec, i, 'recovery_serial', refs, cell='p8192_c8')
                    print('PASS serial pressure/recovery outputs and acceptance', flush=True)
                    for concurrency in (4, 8):
                        for repeat in range(3):
                            ab.dump(root / 'progress.json', {'stage': 'recovery', 'mode': mode,
                                'phase': 'concurrent_pressure', 'concurrency': concurrency, 'repeat': repeat})
                            # Refresh pressure without bypassing any cache guard.
                            for i in range(len(fillers)):
                                await request(session, base, args, server, spec, 7, 'root_keepalive', refs, cell='p8192_c8')
                                await request(session, base, args, server, pressure_spec, i, 'pressure_fill',
                                              pressure_refs, cell='pressure')
                            await wave([request(session, base, args, server, spec, i, 'recovery_concurrent',
                                refs, cell='p8192_c8', tested_concurrency=concurrency, repeat=repeat)
                                for i in range(concurrency)])
                            print(f'PASS recovery c{concurrency} round={repeat}', flush=True)
                    after = ab.info_stats(await ab.api(session, base, '/server_info'))
                    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
                    work = [json.loads(line) for line in (folder / 'worker_work.jsonl').read_text().splitlines()]
                    completed = {str(row['meta_info']['id']): row for row in rows}
                    measured = [w for w in work if w['rid'] in completed]
                    partial = [w for w in measured if w['plan_mode'] == 'partial']
                    full_fallback = [w for w in measured if w['plan_mode'] == 'full' and w['target_hit_tokens'] > 0]
                    result = {'cache_counter_delta': ab.counter_delta(before, after),
                        'sidecar_budget_tokens': budget,
                        'unique_pressure_prefix_tokens': unique_pressure_tokens,
                        'conservative_peak_target_tokens': conservative_peak_target_tokens,
                        'completed_requests': len(rows), 'joined_worker_requests': len(measured),
                        'actual_partial_recovery_requests': len(partial),
                        'actual_recovered_tokens': sum(w['actual_recovered_tokens'] for w in partial),
                        'actual_partial_prefill_tokens': sum(w['actual_prefill_tokens'] for w in partial),
                        'actual_partial_draft_injected_tokens': sum(w['draft_injected_tokens'] for w in partial),
                        'actual_target_only_full_fallback_requests': len(full_fallback),
                        'partial_tested_concurrencies': sorted({completed[w['rid']].get('tested_concurrency', 1) for w in partial}),
                        'scope': 'default 25% sidecar budget, natural LRU pressure; no fault injection; NOT timing A/B'}
                    ab.dump(root / 'recovery_summary.json', result)
                    if len(measured) != len(rows) or not partial:
                        raise ab.SafetyFailure('Worker recovery coverage/join not established')
                    if not {1, 4, 8}.issubset(result['partial_tested_concurrencies']):
                        raise ab.SafetyFailure('Actual partial recovery did not cover serial/4/8')
        finally:
            server.stop()


async def timing_arm(args, dataset, refs, root, block, mode, prior=None):
    import aiohttp
    check_sources(args)
    folder = root / f'block{block:02d}_{mode}'
    folder.mkdir()
    server = QualifiedServer(args, mode, folder)
    gpu = ab.GPUSampler(folder / 'gpu_samples.jsonl', args.gpu_sample_ms)
    summaries = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                         connector=ab.http_connector(aiohttp)) as session:
            gpu.start()
            base = await start(server, session)
            cells = list(dataset)
            random.Random(args.seed + block).shuffle(cells)  # Same within A/B block.
            for cell in cells:
                ab.dump(root / 'progress.json', {'stage': 'ab', 'block': block, 'mode': mode, 'cell': cell})
                spec = dataset[cell]
                await ab.api(session, base, '/flush_cache?timeout=10', 'POST')
                for variant in range(spec['concurrency']):
                    await request(session, base, args, server, spec, variant, 'prime_excluded', refs, cell=cell)
                # Shape-specific thermal/JIT stabilization excluded identically.
                for repeat in range(2):
                    await wave([request(session, base, args, server, spec, v, 'settle_excluded', refs,
                        cell=cell, wave=repeat) for v in range(spec['concurrency'])])
                before = ab.info_stats(await ab.api(session, base, '/server_info'), required=mode == 'paired')
                gpu_start = gpu.begin(cell)
                started = time.perf_counter()
                rows = []
                for repeat in range(args.requests_per_cell // spec['concurrency']):
                    current = await wave([request(session, base, args, server, spec, v, 'steady', refs,
                        cell=cell, wave=repeat, block=block) for v in range(spec['concurrency'])])
                    if mode == 'paired' and any(row['cached_tokens'] != spec['prompt_length'] - 1 for row in current):
                        raise ab.SafetyFailure('Hot A/B window not fully paired; pressure belongs in separate test')
                    if any(row['meta_info'].get('num_retractions') != 0 for row in current):
                        raise ab.SafetyFailure('Fixed hot timing protocol requires zero retractions')
                    rows.extend(current)
                elapsed = time.perf_counter() - started
                after = ab.info_stats(await ab.api(session, base, '/server_info'), required=mode == 'paired')
                item = ab.aggregate(mode, cell, block, rows, before, after, gpu.summary(gpu_start), elapsed)
                delta = item['cache']['counter_delta']
                matches = delta.get('match_calls')
                denominator = matches * (spec['prompt_length'] - 1) if matches else None
                item['cache']['shadow_target_token_rate_per_match'] = (
                    delta.get('target_hit_tokens', 0) / denominator if denominator else None)
                item['cache']['paired_token_rate_per_match'] = (
                    delta.get('paired_hit_tokens', 0) / denominator if denominator else None)
                item['cache']['shadow_rate_scope'] = 'pre-truncation target match observations, not unique-request hit rate'
                item.update(block=block, workload='100% repeated-prefix hot waves; priming/settle excluded',
                    worker_fallback_measurement='unavailable in uninstrumented timing; sidecar deltas are plans')
                ab.append_jsonl(root / 'summaries.jsonl', item)
                summaries.append(item)
                if prior is not None:
                    other = next(summary for summary in prior if summary['cell'] == cell)
                    control, candidate = (item, other) if mode == 'full' else (other, item)
                    ab.check_p95(control, candidate, args.requests_per_cell, 0)
                print(f'PASS steady block={block} {mode} {cell} n={len(rows)}', flush=True)
        return summaries
    finally:
        server.stop()
        gpu.stop()


async def performance(args, dataset, refs, root):
    comparisons = []
    for block, order in enumerate(balanced_order(args.pairs)):
        arms = {}
        for mode in order:
            prior = next(iter(arms.values()), None)
            arms[mode] = await timing_arm(args, dataset, refs, root, block, mode, prior)
        full = {item['cell']: item for item in arms['full']}
        paired = {item['cell']: item for item in arms['paired']}
        for cell in dataset:
            # User's strict stop rule: no tolerance silently added after results.
            ab.check_p95(full[cell], paired[cell], args.requests_per_cell, 0)
            comparison = {'block': block, 'order': list(order), 'cell': cell, 'p95_guard': 'passed',
                'mean_ratios_paired_over_full': {metric: paired[cell][metric]['mean'] / full[cell][metric]['mean']
                    for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}}
            comparisons.append(comparison)
            ab.append_jsonl(root / 'comparisons.jsonl', comparison)
    result = {'comparisons': comparisons, 'performance_claim': 'unproven',
        'scope': 'hot repeated-prefix waves only; no mixed/cold or safe target-only arm',
        'promotion': 'Requires balanced blocks, no p95 regression, and CI-backed material gain'}
    for cell, spec in dataset.items():
        left, right = [], []
        for block in range(args.pairs):
            for mode, destination in [('full', left), ('paired', right)]:
                rows = [json.loads(line) for line in (root / f'block{block:02d}_{mode}' / 'requests.jsonl').read_text().splitlines()]
                waves = {}
                for row in rows:
                    if row['cell'] == cell and row['phase'] == 'steady':
                        waves.setdefault(row['wave'], []).append(row['ttft_ms'])
                destination.extend(sum(waves[key]) / len(waves[key]) for key in sorted(waves))
        result.setdefault('ttft_cluster_bootstrap', {})[cell] = bootstrap_ratio(left, right, args.seed)
    intervals = list(result['ttft_cluster_bootstrap'].values())
    if all(item is not None and item['ci95_high'] < .95 for item in intervals):
        result['performance_claim'] = 'material hot-TTFT gain demonstrated for tested cells; not general deployment proof'
    ab.dump(root / 'performance_summary.json', result)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['recovery', 'ab'], required=True)
    parser.add_argument('--cells', default='all')
    parser.add_argument('--requests-per-cell', type=int, default=64)
    parser.add_argument('--pairs', type=int, default=2)
    cli = parser.parse_args()
    manifest = json.loads((cli.qualified / 'manifest.json').read_text())
    qualified = json.loads((cli.qualified / 'status.json').read_text())
    if (qualified.get('status') != 'passed_for_requested_cells_only'
            or not qualified.get('full_correctness_passed') or not qualified.get('paired_correctness_passed')
            or not qualified.get('old_references_unchanged') or not qualified.get('inference_sources_unchanged')):
        raise ValueError('Both qualified correctness gates must pass first')
    if (manifest['profile']['name'] != 'dspark-stable-fa4-fixed64-v2'
            or manifest['profile'].get('fixed_fa4_tile') != [64, 64]
            or manifest['profile_id'] != qualified['profile_id']):
        raise ValueError('Requires the qualified fixed64 v2 profile identity')
    numerical = {name: manifest[name] for name in ('profile', 'source_sha256', 'packages',
        'fixed_policy_sha256', 'model_metadata_sha256', 'sps_sha256', 'gpu_identity')}
    digest = hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()[:24]
    if manifest['profile_id'] != manifest['profile']['name'] + '-' + digest:
        raise ValueError('Qualified numerical profile digest does not match its manifest')
    if sha(cli.qualified / 'baseline_references.jsonl') != qualified['baseline_reference_sha256']:
        raise ValueError('Qualified reference digest changed')
    if manifest['args']['output_tokens'] < 128:
        raise ValueError('Requires long-output qualified profile')
    args = SimpleNamespace(**manifest['args'])
    args.output = str(cli.output)
    args.old_oracle_jsonl = args.oracle_jsonl
    args.oracle_jsonl = str(cli.qualified / 'baseline_references.jsonl')
    args.profile_id = manifest['profile_id']
    args.source_sha256 = manifest['source_sha256']
    args.observe_work = cli.stage == 'recovery'
    args.requests_per_cell, args.pairs = cli.requests_per_cell, cli.pairs
    args.warmup_prompt = json.loads((cli.qualified / 'dataset.json').read_text())['p512_c1']['prompts'][0]
    if args.cache_budget_fraction != .25:
        raise ValueError('Protocol requires original 25% sidecar budget')
    if args.requests_per_cell < 64 or args.requests_per_cell % 8:
        raise ValueError('At least 64 requests/cell, divisible by eight')
    balanced_order(args.pairs)
    repo = Path(args.repo)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != cli.expected_head or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
        raise ValueError('Requires pinned clean experimental commit')
    check_sources(args)
    execution_paths = ('python/sglang/srt/managers/schedule_batch.py',
        'python/sglang/srt/mem_cache/dspark_prefix_cache.py',
        'python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py')
    args.execution_source_sha256 = {}
    for path in execution_paths:
        original = subprocess.check_output(['git', 'show', manifest['head'] + ':' + path], cwd=repo)
        digest = hashlib.sha256(original).hexdigest()
        if sha(repo / path) != digest:
            raise ValueError('Qualified scheduler/cache/worker execution source changed: ' + path)
        args.execution_source_sha256[path] = digest
    # Pre-existing UNTRACKED dependency: preserve it; do NOT git-add it. This
    # additional digest is frozen now, not misrepresented as an old gate digest.
    untracked = 'python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py'
    args.preserved_untracked_source_sha256 = {untracked: sha(repo / untracked)}
    versions = {name: importlib.metadata.version(name) for name in manifest['packages']}
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                       '--format=csv,noheader'], text=True).strip()
    if versions != manifest['packages'] or identity != manifest['gpu_identity']:
        raise ValueError('Qualified packages/GPU changed')
    for directory, expected in manifest['model_metadata_sha256'].items():
        actual = {path.name: sha(path) for path in Path(directory).glob('*.json')}
        if actual != expected:
            raise ValueError('Qualified model metadata changed')
    if sha(args.sps_table) != manifest['sps_sha256']:
        raise ValueError('Qualified SPS table changed')
    if sha(Path(__file__).with_name('fixed_fa4_policy_hook.py')) != manifest['fixed_policy_sha256']:
        raise ValueError('Qualified fixed numerical policy changed')
    full_data = json.loads((cli.qualified / 'dataset.json').read_text())
    refs = ab.load_frozen_references(cli.qualified / 'baseline_references.jsonl', full_data)
    cells = list(full_data) if cli.cells == 'all' else cli.cells.split(',')
    if len(cells) != len(set(cells)) or any(cell not in full_data for cell in cells):
        raise ValueError('Unknown/duplicate cells')
    dataset = {cell: full_data[cell] for cell in cells}
    frozen = {str(path): sha(path) for path in (cli.qualified / 'baseline_references.jsonl', cli.qualified / 'dataset.json')}
    frozen.update(manifest['old_references_read_only'])
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError('Historical reference integrity failed')
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / 'dataset.json', full_data)
    ab.dump(cli.output / 'manifest.json', {'args': vars(args), 'head': head, 'stage': cli.stage,
        'qualified_manifest': manifest, 'frozen': frozen, 'cells': cells,
        'prespecified': {'order': balanced_order(args.pairs), 'requests_per_cell_per_arm_block': args.requests_per_cell,
            'p95_regression_tolerance': 0, 'material_ttft_gain_threshold': .05,
            'confidence_method': 'paired-wave cluster bootstrap; descriptive p95; limited independent blocks',
            'stop_on_tokens_or_any_five_acceptance_changes': True}, 'started_at': time.time()})
    status = {'status': 'running', 'performance_claim': 'unproven'}
    try:
        if cli.stage == 'recovery':
            await recovery(args, full_data, refs, cli.output)
        else:
            await performance(args, dataset, refs, cli.output)
        status['status'] = 'passed_for_requested_stage_only'
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            check_sources(args)
            status['execution_sources_unchanged'] = True
            status['references_unchanged'] = all(sha(path) == digest for path, digest in frozen.items())
        except Exception as exc:
            status.update(status='safety_stopped', execution_sources_unchanged=False,
                          error_type=type(exc).__name__, error=str(exc))
        status['finished_at'] = time.time()
        if status.get('references_unchanged') is False:
            status.update(status='safety_stopped', error='Frozen reference integrity failed')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'passed_for_requested_stage_only' else 2


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
