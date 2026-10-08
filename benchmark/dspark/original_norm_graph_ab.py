"""One isolated non-GEMM candidate, unprofiled A/B then B/A, immutable oracle.

No GEMM/FA4/reduction/KV/cache/planner/sampling changes. Only original norm subgraphs. Any correctness,
acceptance, p95, noise, or >=1% benefit failure is terminal; no selective rerun.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import aiohttp
import dspark_prefix_ab as ab
from graph_diagnostic import GraphServer
from stable_baseline_gate import sha, validate_row
from stable_cache_validation import start, wave
from steady_tpot import DeviceTelemetry, admitted, drain_listener, integrity
from verify_all_ab import noise_gate


PROTOCOL = {
    'timing_cell': 'p2048_c1', 'output_tokens': 128,
    'correctness_cells': ['p2048_c1', 'p2048_c4', 'p2048_c8'],
    'correctness_every_arm': 'all serial variants then two concurrent waves: 39 requests',
    'warmup_min': 24, 'warmup_max': 64, 'warmup_group': 8,
    'warmup_last_three_tpot_mean_spread': .005, 'warmup_p95_over_p50': 1.02,
    'control_windows': 2, 'steady_requests_per_window': 64,
    'control_tpot_and_e2e_mean_spread': .005, 'control_tpot_and_e2e_p95_spread': .01,
    'candidate_order': ['baseline', 'graph', 'graph', 'baseline'],
    'candidate_p95_tolerance_ttft_tpot_e2e': 0.0,
    'minimum_e2e_gain_each_order': .01,
    'profiler': False, 'diagnostic_hooks': False, 'request_retries': 0,
    'stop_on_any_token_or_five_acceptance_difference': True,
    'cache': 'full prefill in both arms', 'gemm': 'unchanged', 'mean': 'unchanged',
    'planner': 'unchanged original SPS/confidence', 'framework_graphs': 'OFF', 'norm_subgraphs': 'exact shape, original arithmetic',
    'performance_claim_scope': 'only p2048 c1 output128; not matrix/deployment',
    'wall_budget_including_startup_warmup_shutdown_seconds': 2400,
}


def tool_digests():
    return {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}


def validate_graph_attestation(folder, candidate):
    records = list((folder / 'graph_policy').glob('graph_policy.json'))
    if len(records) != (1 if candidate else 0):
        raise ab.SafetyFailure('Unexpected graph hook installation')
    if candidate:
        record = json.loads(records[0].read_text())
        if not record['enabled'] or record['diagnostic'] or not record['original_arithmetic']:
            raise ab.SafetyFailure('Graph runtime attestation failed')


async def arm(args, template, refs, dataset, root, ordinal, variant, windows):
    integrity(template, args)
    folder = root / f'{ordinal:02d}_{variant}'
    folder.mkdir()
    args.output = str(folder)
    candidate = variant == 'graph'
    server = GraphServer(args, 'full', folder, diagnostic=False, candidate=candidate)
    telemetry = DeviceTelemetry(folder)
    summaries = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                connector=ab.http_connector(aiohttp)) as session:
            telemetry.begin()
            base = await start(server, session)
            validate_graph_attestation(folder, candidate)

            async def checked(cell, variant_id, phase, number, window=None):
                spec = dataset[cell]
                began = time.time()
                row = await ab.generate(session, base, spec['prompts'][variant_id], 128, args.seed)
                row.update(cell=cell, variant=variant_id, concurrency=spec['concurrency'],
                    phase=phase, ordinal=number, window=window, kernel_variant=variant,
                    arm=ordinal, began_wall_time=began, finished_wall_time=time.time())
                ab.append_jsonl(folder / 'requests.jsonl', row)
                validate_row(row, spec, 128, ordinary=False, require_cold=True)
                ab.check_reference(refs[(cell, variant_id)], row)
                if row['meta_info'].get('num_retractions') != 0:
                    raise ab.SafetyFailure('Requires zero retractions')
                server.check()
                return row

            for cell in PROTOCOL['correctness_cells']:
                spec = dataset[cell]
                for v in range(spec['concurrency']):
                    await checked(cell, v, 'correctness_serial', v)
                for repeat in range(2):
                    await wave([checked(cell, v, 'correctness_concurrent', repeat)
                                for v in range(spec['concurrency'])])
                print(f'CORRECTNESS {folder.name} {cell} serial+2waves PASS', flush=True)
            ab.dump(folder / 'correctness.json', {'passed': True, 'requests': 39,
                'frozen_oracle': args.oracle_jsonl, 'tokens_and_five_acceptance': True,
                'cells': PROTOCOL['correctness_cells']})

            await checked('p2048_c1', 0, 'prime_excluded', 0)
            groups = []
            for group in range(PROTOCOL['warmup_max'] // 8):
                rows = [await checked('p2048_c1', 0, 'warmup_excluded', group * 8 + i)
                        for i in range(8)]
                item = ab.distribution([row['tpot_ms'] for row in rows])
                groups.append(item)
                ab.append_jsonl(folder / 'warmup_groups.jsonl', {'group': group, **item})
                print('WARMUP ' + folder.name + ' ' + json.dumps(item), flush=True)
                if admitted(groups):
                    break
            else:
                raise ab.SafetyFailure('Warmup admission failed; no retry/selective discard')
            ab.dump(folder / 'warmup_admission.json', {'groups': groups, 'admitted': True})
            if candidate:
                ab.dump(folder / 'graph_policy' / 'seal_catalog', {'excluded': True})
                await checked('p2048_c1', 0, 'seal_excluded', 0)
                sealed = json.loads((folder / 'graph_policy' / 'catalog_sealed.json').read_text())
                if not sealed['sealed'] or not sealed['captures'] or not sealed['replays']:
                    raise ab.SafetyFailure('No sealed/replaying original norm graph catalog')

            for window in range(windows):
                before = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                began = time.perf_counter()
                rows = []
                for i in range(64):
                    rows.append(await checked('p2048_c1', 0, 'steady', i, window))
                    if (i + 1) % 8 == 0:
                        print(f'STEADY {folder.name} w{window} n{i+1}', flush=True)
                after = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                item = ab.aggregate('full', 'p2048_c1', window, rows, before, after,
                    {'scope': 'device_telemetry.csv; device-wide, not allocator peak'},
                    time.perf_counter() - began)
                item.update(arm=ordinal, window=window, kernel_variant=variant,
                            profiler=False, diagnostic_hooks=False)
                ab.append_jsonl(root / 'summaries.jsonl', item)
                summaries.append(item)
                print('WINDOW ' + folder.name + ' ' + json.dumps({key: item[key] for key in
                    ('ttft_ms', 'tpot_ms', 'e2e_ms', 'acceptance')}), flush=True)

            if candidate:
                ab.dump(folder / 'graph_policy' / 'snapshot_catalog', {'excluded': True})
                await checked('p2048_c1', 0, 'snapshot_excluded', 0)
                final = json.loads((folder / 'graph_policy' / 'catalog_final.json').read_text())
                if final['captures'] != sealed['captures'] or final['replays'] <= sealed['replays']:
                    raise ab.SafetyFailure('Capture during steady state or no graph replays')
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
    integrity(template, args)
    return summaries


async def run(cli):
    inherited = json.loads((cli.inherit / 'manifest.json').read_text())
    template = copy.deepcopy(inherited['execution_integrity_template'])
    args = SimpleNamespace(**template['args'])
    args.observe_work = False
    args.startup_timeout = min(args.startup_timeout, 600)
    dataset = json.loads((Path(args.oracle_jsonl).parent / 'dataset.json').read_text())
    refs = ab.load_frozen_references(Path(args.oracle_jsonl), dataset)
    # Newly pinned preserved dependency, not misrepresented as an old gate digest.
    layernorm = Path(args.repo) / 'python/sglang/srt/layers/layernorm.py'
    template['frozen'][str(layernorm)] = sha(layernorm)
    tools = tool_digests()
    integrity(template, args)
    cli.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    ab.dump(cli.output / 'manifest.json', {'protocol': PROTOCOL, 'created_at': began,
        'source': str(cli.inherit), 'source_sha256': sha(cli.inherit / 'manifest.json'),
        'execution_integrity_template': template, 'tool_sha256': tools,
        'experiment_worktree_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent, text=True).strip(),
        'layernorm_dependency_sha256_pinned_now': sha(layernorm), 'performance': 'UNPROVEN'})
    status = {'status': 'running', 'performance_claim': 'unproven'}
    ab.dump(cli.output / 'status.json', status)

    async def execute():
        controls = await arm(args, template, refs, dataset, cli.output, 0, 'baseline', 2)
        control = noise_gate(*controls)
        ab.dump(cli.output / 'control_noise.json', control)
        if not control['admitted']:
            raise ab.SafetyFailure('Baseline/baseline noise gate failed; no candidate timing')
        status['baseline_baseline_noise_admitted'] = True
        arms, comparisons = {}, []
        for ordinal, variant in enumerate(PROTOCOL['candidate_order'], 1):
            arms[ordinal] = (await arm(args, template, refs, dataset, cli.output,
                                      ordinal, variant, 1))[0]
            if ordinal in (2, 4):
                control, candidate = ((arms[1], arms[2]) if ordinal == 2 else (arms[4], arms[3]))
                ratios = {metric: {stat: candidate[metric][stat] / control[metric][stat]
                            for stat in ('mean', 'p95')}
                          for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
                item = {'order': 'AB' if ordinal == 2 else 'BA',
                        'ratios_graph_over_baseline': ratios,
                        'e2e_gain_at_least_one_percent': ratios['e2e_ms']['mean'] <= .99}
                # Preserve ratios even when the prespecified p95 guard rejects them.
                ab.append_jsonl(cli.output / 'comparisons.jsonl', item)
                if ab.check_p95(control, candidate, 64, 0.0) != 'passed':
                    raise ab.SafetyFailure('Insufficient samples for p95 gate')
                comparisons.append(item)
                if not item['e2e_gain_at_least_one_percent']:
                    status.update(status='below_benefit_threshold_stopped',
                        stop_reason='An order failed >=1% mean E2E; remaining orders not run')
                    return
        drift = noise_gate(arms[1], arms[4])
        ab.dump(cli.output / 'counterbalanced_control_drift.json', drift)
        if not drift['admitted']:
            raise ab.SafetyFailure('Counterbalanced control drift exceeds noise gate')
        status.update(status='counterbalanced_one_cell_completed',
            performance_claim='>=1% mean E2E BOTH orders; tested cell only; no p95 regression')

    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await asyncio.wait_for(execute(), timeout=max(0, began + 2400 - time.time()))
    except asyncio.CancelledError:
        status.update(status='safety_stopped', error_type='CancelledError', error='Supervisor stop')
    except Exception as error:
        status.update(status='safety_stopped', error_type=type(error).__name__, error=str(error))
    finally:
        try:
            integrity(template, args)
            if tools != tool_digests():
                raise ab.SafetyFailure('Experiment tool sources changed during run')
            status['sources_and_frozen_references_unchanged'] = True
        except Exception as error:
            status.update(status='safety_stopped', sources_and_frozen_references_unchanged=False,
                          error_type=type(error).__name__, error=str(error))
        status.update(finished_at=time.time(),
            process_local_rollback='owned graph-catalog service stopped; serving baseline source untouched')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 2 if status['status'] == 'safety_stopped' else 3 if status['status'].endswith('_stopped') else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
