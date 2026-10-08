"""Bounded, one-candidate unprofiled fixed-policy A/B against immutable oracle.

Hardware migration changes ONLY GPU UUID, explicitly recorded before serving;
same name/driver/packages/sources/weights metadata required. No new oracle, no
retrying failed control admission, no paired-cache/GEMM/graph/quant experiments.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import sha, validate_row
from stable_cache_validation import QualifiedServer, start, wave
from steady_tpot import DeviceTelemetry, admitted, drain_listener, integrity
from verify_all_policy_hook import POLICY

PROTOCOL = {'timing_cell': 'p2048_c1', 'output_tokens': 128,
    'correctness_cells': ['p2048_c1', 'p2048_c4', 'p2048_c8'],
    'correctness_serial_variants_and_concurrent_rounds': 2,
    'identical_correctness_preconditioning_every_arm': True,
    'warmup_min': 24, 'warmup_max': 64, 'warmup_group': 8,
    'warmup_last_three_tpot_mean_spread': .005, 'warmup_p95_over_p50': 1.02,
    'control_windows': 2, 'steady_requests_per_window': 64,
    'control_tpot_and_e2e_mean_spread': .005, 'control_tpot_and_e2e_p95_spread': .01,
    'candidate_order': ['adaptive', 'fixed', 'fixed', 'adaptive'],
    'candidate_p95_tolerance_ttft_tpot_e2e': 0.0,
    'minimum_e2e_gain_each_order': .01, 'profiler': False, 'request_retries': 0,
    'stop_on_any_token_or_five_acceptance_difference': True,
    'performance_claim_scope': 'only p2048 c1 output128; not full matrix/deployment',
    'wall_budget_including_startup_warmup_shutdown_seconds': 7200}

EXTRA_INPUTS = ('python/sglang/srt/speculative/dspark_components/dspark_planner.py',
    'python/sglang/srt/speculative/dspark_components/dspark_draft.py',
    'python/sglang/srt/speculative/dspark_components/dspark_verify.py',
    'python/sglang/srt/speculative/dspark_components/dspark_observability.py',
    'python/sglang/srt/managers/overlap_utils.py',
    'python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py',
    'python/sglang/srt/speculative/ragged_verify.py')


def migrated_template(original, current_gpu):
    old = original['qualified_manifest']['gpu_identity'].split(', ')
    new = current_gpu.split(', ')
    if len(old) != 3 or len(new) != 3 or (old[0], old[2]) != (new[0], new[2]):
        raise ab.SafetyFailure('Migration permits UUID only; different model/driver needs separate qualification')
    result = copy.deepcopy(original)
    result['qualified_manifest']['gpu_identity'] = current_gpu
    return result


def noise_gate(left, right):
    result = {}
    for metric in ('tpot_ms', 'e2e_ms'):
        means, tails = ([left[metric][key], right[metric][key]] for key in ('mean', 'p95'))
        result[metric] = {'mean_spread': max(means) / min(means) - 1,
                         'p95_spread': max(tails) / min(tails) - 1}
    result['admitted'] = all(item['mean_spread'] <= .005 and item['p95_spread'] <= .01
                            for item in result.values())
    return result


def comparison(control, candidate):
    if ab.check_p95(control, candidate, 64, 0.0) != 'passed':
        raise ab.SafetyFailure('Insufficient samples for prespecified p95 gate')
    ratios = {metric: {stat: candidate[metric][stat] / control[metric][stat]
                        for stat in ('mean', 'p95')}
              for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    return {'ratios_fixed_over_adaptive': ratios, 'p95_guard': 'passed',
            'e2e_gain_at_least_one_percent': ratios['e2e_ms']['mean'] <= .99}


class PolicyServer(QualifiedServer):
    def __init__(self, args, variant, folder):
        # Both arms are full-prefill, not cache-policy comparisons.
        super().__init__(args, 'full', folder)
        self.variant = variant

    def command(self):
        cmd, env = super().command()
        env['SGLANG_DSPARK_FIXED_VERIFY_ALL_EXPERIMENT'] = '0'
        if self.variant == 'fixed':
            index = cmd.index('--speculative-dspark-sps-table-path')
            del cmd[index:index + 2]
            index = cmd.index('--forward-hooks') + 1
            hooks = json.loads(cmd[index])
            hooks[0]['hook_factory'] = 'verify_all_policy_hook:make_hook'
            hooks[0]['config'].update(verify_policy=POLICY,
                                      policy_folder=str(self.folder / 'policy'))
            cmd[index] = json.dumps(hooks)
            env['SGLANG_DSPARK_FIXED_VERIFY_ALL_EXPERIMENT'] = '1'
        return cmd, env


def validated_startup_resume(root, current_template, refs):
    """Resume only a pre-measurement attestation bug, never failed performance.

    Controls stay immutable, all records are rechecked, and no control window is
    repeated. A correctness/p95/warmup/noise stop is NOT resumable here.
    """
    previous = json.loads((root / 'manifest.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    if (status.get('error') != 'Fixed policy did not initialize exactly one planner' or
            not status.get('hardware_migration_correctness_passed') or
            not status.get('full_full_noise_admitted') or
            previous['protocol'] != PROTOCOL or
            previous['execution_integrity_template'] != current_template or
            (root / '02_fixed' / 'requests.jsonl').exists() or
            (root / '03_fixed').exists()):
        raise ab.SafetyFailure('Cannot resume a measured/failed correctness or performance candidate')
    summaries = [json.loads(line) for line in (root / 'summaries.jsonl').read_text().splitlines()]
    if [(r['arm'], r['window']) for r in summaries] != [(0, 0), (0, 1), (1, 0)]:
        raise ab.SafetyFailure('Resume requires the exact original completed control windows')
    for name in ('00_adaptive', '01_adaptive'):
        rows = [json.loads(line) for line in (root / name / 'requests.jsonl').read_text().splitlines()]
        for row in rows:
            ab.check_reference(refs[(row['cell'], row['variant'])], row)
            if row['cached_tokens'] != 0 or row['meta_info'].get('num_retractions') != 0:
                raise ab.SafetyFailure('Inherited control geometry/retractions changed')
        if sum(r['phase'] == 'steady' for r in rows) != (128 if name == '00_adaptive' else 64):
            raise ab.SafetyFailure('Incomplete inherited control')
    digests = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
    return summaries, digests


async def arm(args, template, refs, dataset, root, ordinal, variant, windows,
              correctness=False):
    import aiohttp
    integrity(template, args)
    folder = root / f'{ordinal:02d}_{variant}'
    folder.mkdir()
    args.output = str(folder)
    server, telemetry = PolicyServer(args, variant, folder), DeviceTelemetry(folder)
    summaries = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                         connector=ab.http_connector(aiohttp)) as session:
            telemetry.begin()
            base = await start(server, session)
            if variant == 'fixed':
                records = list((folder / 'policy').glob('planner-*.json'))
                if len(records) != 1:
                    raise ab.SafetyFailure('Fixed policy did not initialize exactly one planner')
                data = json.loads(records[0].read_text())
                if (data['policy'] != POLICY or data['host_budget_enabled'] or
                        not data['confidence_head_preserved']):
                    raise ab.SafetyFailure('Fixed policy runtime attestation failed')

            async def checked(cell, variant_id, phase, number, window=None):
                spec = dataset[cell]
                began = time.time()
                row = await ab.generate(session, base, spec['prompts'][variant_id], 128, args.seed)
                row.update(cell=cell, variant=variant_id, concurrency=spec['concurrency'],
                    phase=phase, ordinal=number, window=window, policy_variant=variant,
                    arm=ordinal, began_wall_time=began, finished_wall_time=time.time())
                ab.append_jsonl(folder / 'requests.jsonl', row)
                validate_row(row, spec, 128, ordinary=False, require_cold=True)
                ab.check_reference(refs[(cell, variant_id)], row)
                if row['meta_info'].get('num_retractions') != 0:
                    raise ab.SafetyFailure('Requires zero retractions')
                server.check()
                return row

            if correctness:
                for cell in PROTOCOL['correctness_cells']:
                    spec = dataset[cell]
                    for v in range(spec['concurrency']):
                        await checked(cell, v, 'migration_or_policy_serial', v)
                    for repeat in range(2):
                        await wave([checked(cell, v, 'correctness_concurrent', repeat)
                                    for v in range(spec['concurrency'])])
                    print(f'CORRECTNESS {variant} {cell} serial+2waves PASS', flush=True)
                ab.dump(folder / 'correctness.json', {'passed': True, 'requests': 39,
                    'frozen_oracle': args.oracle_jsonl, 'tokens_and_five_acceptance': True,
                    'cells': PROTOCOL['correctness_cells']})

            await checked('p2048_c1', 0, 'prime_excluded', 0)
            groups = []
            for group in range(8):
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
                item.update(arm=ordinal, window=window, policy_variant=variant,
                            profiler=False, diagnostic_hooks=False)
                ab.append_jsonl(root / 'summaries.jsonl', item)
                summaries.append(item)
                print('WINDOW ' + folder.name + ' ' + json.dumps({key: item[key] for key in
                    ('ttft_ms', 'tpot_ms', 'e2e_ms', 'acceptance')}), flush=True)
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
    integrity(template, args)
    return summaries


async def run(cli):
    original = json.loads((cli.template / 'manifest.json').read_text())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'], text=True).strip()
    template = migrated_template(original, gpu)
    args = SimpleNamespace(**template['args'])
    args.observe_work = False
    args.startup_timeout = min(args.startup_timeout, 600)
    # Newly pinned dependency evidence is not misrepresented as historical SHA.
    extra = {path: sha(Path(args.repo) / path) for path in EXTRA_INPUTS}
    template['frozen'].update(extra)
    integrity(template, args)
    dataset = json.loads((cli.template / 'dataset.json').read_text())
    refs = ab.load_frozen_references(Path(args.oracle_jsonl), dataset)
    inherited = validated_startup_resume(cli.resume_startup_repair, template, refs) if cli.resume_startup_repair else None
    tools = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    budget_origin = (json.loads(cli.budget_origin.read_text())['created_at']
                     if cli.budget_origin else time.time())
    if budget_origin > time.time():
        raise ValueError('Budget origin cannot move the deadline into the future')
    manifest = {'protocol': PROTOCOL, 'policy': POLICY,
        'original_template_sha256': sha(cli.template / 'manifest.json'),
        'original_qualified_gpu': original['qualified_manifest']['gpu_identity'],
        'current_gpu': gpu, 'hardware_migration': 'UUID only; qualification pending new full oracle checks',
        'inherited_arithmetic_contract_id': args.profile_id,
        'execution_integrity_template': template, 'extra_sources_pinned_now': extra,
        'tool_sha256': tools, 'created_at': time.time(), 'performance_claim': 'unproven',
        'wall_budget_origin': budget_origin, 'wall_deadline': budget_origin + 7200,
        'prior_aborted_record': str(cli.budget_origin) if cli.budget_origin else None,
        'inherited_controls_from_startup_repair': str(cli.resume_startup_repair) if inherited else None,
        'inherited_read_only_sha256': inherited[1] if inherited else {}}
    manifest['experiment_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / 'manifest.json', manifest)
    status = {'status': 'running', 'performance_claim': 'unproven',
              'experiment_id': manifest['experiment_id']}

    async def execute():
        if inherited:
            full = inherited[0][:2]
            for name in ('00_adaptive', '01_adaptive'):
                (cli.output / name).symlink_to((cli.resume_startup_repair / name).resolve(), target_is_directory=True)
            for row in inherited[0]:
                ab.append_jsonl(cli.output / 'summaries.jsonl', row)
        else:
            full = await arm(args, template, refs, dataset, cli.output, 0, 'adaptive', 2, True)
        status['hardware_migration_correctness_passed'] = True
        control = noise_gate(*full)
        ab.dump(cli.output / 'control_noise.json', control)
        if not control['admitted']:
            raise ab.SafetyFailure('Full/full noise admission failed; candidate not started')
        status['full_full_noise_admitted'] = True
        arms, comparisons = ({1: inherited[0][2]} if inherited else {}), []
        for ordinal, variant in enumerate(PROTOCOL['candidate_order'], 1):
            if inherited and ordinal == 1:
                continue
            arms[ordinal] = (await arm(args, template, refs, dataset, cli.output,
                ordinal, variant, 1, correctness=True))[0]
            if ordinal in (2, 4):
                control, candidate = ((arms[1], arms[2]) if ordinal == 2 else (arms[4], arms[3]))
                item = comparison(control, candidate)
                item['order'] = 'AB' if ordinal == 2 else 'BA'
                comparisons.append(item)
                ab.append_jsonl(cli.output / 'comparisons.jsonl', item)
        drift = noise_gate(arms[1], arms[4])
        ab.dump(cli.output / 'counterbalanced_control_drift.json', drift)
        if not drift['admitted']:
            raise ab.SafetyFailure('AB/BA control drift exceeds prespecified noise limits')
        status['status'] = 'counterbalanced_one_cell_completed'
        if all(item['e2e_gain_at_least_one_percent'] for item in comparisons):
            status['performance_claim'] = '>=1% mean E2E in BOTH orders; tested cell only; no p95 regression'

    # Catch supervisor SIGTERM and run the owned server/telemetry cleanup.
    current_task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, current_task.cancel)
    try:
        remaining = max(0, budget_origin + 7200 - time.time())
        await asyncio.wait_for(execute(), timeout=remaining)
    except asyncio.CancelledError:
        status.update(status='safety_stopped', error_type='CancelledError',
                      error='Supervisor stop; owned services cleaned up')
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            integrity(template, args)
            if inherited and any(sha(path) != digest for path, digest in inherited[1].items()):
                raise ab.SafetyFailure('Inherited control/startup evidence changed')
            status['sources_and_frozen_references_unchanged'] = True
        except Exception as exc:
            status.update(status='safety_stopped', sources_and_frozen_references_unchanged=False,
                          error_type=type(exc).__name__, error=str(exc))
        status.update(finished_at=time.time(), process_local_policy_rollback='owned service stopped; baseline source untouched')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 2 if status['status'] == 'safety_stopped' else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-origin', type=Path,
                        help='Original pre-result manifest; repairs cannot reset wall budget')
    parser.add_argument('--resume-startup-repair', type=Path,
                        help='Only unmeasured planner-attestation failure; immutable controls reused')
    raise SystemExit(asyncio.run(run(parser.parse_args())))
