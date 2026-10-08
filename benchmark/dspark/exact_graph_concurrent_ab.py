"""Unprofiled c4/c8 extension of the unchanged h58 exact-1x8 candidate.

Closed-loop concurrency waves, not open-loop saturation or graph expansion.
Preserve every response; fail closed on frozen tokens/five acceptance or p95.
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
from exact_target_graph_ab import validate_graph_attestation
from target_graph_diagnostic import TargetGraphServer, start
from stable_baseline_gate import sha, validate_row
from stable_cache_validation import wave, bootstrap_ratio
from steady_tpot import DeviceTelemetry, admitted, drain_listener, integrity
from verify_all_ab import noise_gate


PROTOCOL = {
    'cells': ['p2048_c4', 'p2048_c8'], 'output_tokens': 128,
    'correctness_cells': ['p2048_c1', 'p2048_c4', 'p2048_c8'],
    'correctness_serial_and_two_concurrent_waves_every_arm': 39,
    'load': 'closed-loop barrier waves; distinct frozen variants simultaneously',
    'steady_waves_per_window_per_cell': 16,
    'steady_requests_c4': 64, 'steady_requests_c8': 128,
    'warmup_group_waves': 2, 'warmup_min_groups': 3, 'warmup_max_groups': 8,
    'warmup_gate_unit': 'wave mean TPOT (not heterogeneous within-wave requests)',
    'warmup_mean_spread_limit': .005, 'warmup_wave_p95_over_p50_limit': 1.02,
    'control_windows': 2, 'control_mean_spread_limit': .005,
    'control_p95_spread_limit': .01,
    'order': ['baseline', 'target_graph', 'target_graph', 'baseline'],
    'candidate_p95_tolerance': 0.0, 'minimum_e2e_gain_each_order': .01,
    'stop_immediately_after_first_failing_paired_block': True,
    'below_one_percent_is_reported_not_a_reason_to_drop_BA': True,
    'request_retries': 0, 'profiler': False, 'diagnostic_hooks': False,
    'caches': 'OFF; full prefill in both arms',
    'candidate': 'unchanged h58 strict 1x8 + CURRENT device-length host bridge',
    'graph_scope': 'bs1/[8] only; multirow remains eager; no new capture geometry',
    'claim_scope': '2K/128 c4,c8 closed-loop waves on current instance only',
    'wall_budget_seconds': 3600,
}


def tools_sha():
    return {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}


def batch_labels(cell, number, variants):
    if variants != list(range(int(cell.rsplit('c', 1)[1]))):
        raise ValueError('Wave must contain every frozen variant exactly once')
    return {'cell': cell, 'wave': number, 'variants': variants}


def ratios(control, candidate):
    return {m: {s: candidate[m][s] / control[m][s] for s in ('mean', 'p95')}
            for m in ('ttft_ms', 'tpot_ms', 'e2e_ms')}


async def arm(args, template, refs, dataset, root, ordinal, variant, windows):
    integrity(template, args)
    folder = root / f'{ordinal:02d}_{variant}'
    folder.mkdir()
    args.output = str(folder)
    candidate = variant == 'target_graph'
    server = TargetGraphServer(args, 'full', folder, diagnostic=False, candidate=candidate)
    telemetry = DeviceTelemetry(folder)
    summaries = {}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                connector=ab.http_connector(aiohttp)) as session:
            telemetry.begin()
            base = await start(server, session)
            validate_graph_attestation(folder, candidate)

            async def checked(cell, v, phase, number, window=None):
                began = time.time()
                spec = dataset[cell]
                row = await ab.generate(session, base, spec['prompts'][v], 128, args.seed)
                row.update(cell=cell, variant=v, concurrency=spec['concurrency'], phase=phase,
                    wave=number, window=window, kernel_variant=variant, arm=ordinal,
                    began_wall_time=began, finished_wall_time=time.time())
                ab.append_jsonl(folder/'requests.jsonl', row)
                validate_row(row, spec, 128, ordinary=False, require_cold=True)
                ab.check_reference(refs[(cell, v)], row)
                if row['meta_info'].get('num_retractions') != 0:
                    raise ab.SafetyFailure('Requires zero retractions')
                server.check()
                return row

            async def checked_wave(cell, phase, number, window=None):
                count = dataset[cell]['concurrency']
                labels = batch_labels(cell, number, list(range(count)))
                began = time.perf_counter()
                rows = await wave([checked(cell, v, phase, number, window) for v in range(count)])
                elapsed = time.perf_counter() - began
                ab.append_jsonl(folder/'waves.jsonl', {**labels, 'phase': phase, 'window': window,
                    'elapsed_s': elapsed, 'request_count': len(rows),
                    'mean_tpot_ms': sum(r['tpot_ms'] for r in rows)/len(rows),
                    'mean_e2e_ms': sum(r['e2e_ms'] for r in rows)/len(rows),
                    'output_tokens_per_s': count*128/elapsed})
                return rows

            for cell in PROTOCOL['correctness_cells']:
                for v in range(dataset[cell]['concurrency']):
                    await checked(cell, v, 'correctness_serial', v)
                for repeat in range(2):
                    await checked_wave(cell, 'correctness_concurrent', repeat)
                print(f'CORRECTNESS {folder.name} {cell} serial+2waves PASS', flush=True)
            ab.dump(folder/'correctness.json', {'passed': True, 'requests': 39,
                'frozen_oracle': args.oracle_jsonl, 'tokens_and_five_acceptance': True})

            # Admit each load separately. Wave means avoid mistaking fixed
            # prompt/acceptance heterogeneity for temporal warmup instability.
            for cell in PROTOCOL['cells']:
                await checked_wave(cell, 'prime_excluded', 0)
                groups = []
                for group in range(PROTOCOL['warmup_max_groups']):
                    wave_means = []
                    for i in range(PROTOCOL['warmup_group_waves']):
                        rows = await checked_wave(cell, 'warmup_excluded', group*2+i)
                        wave_means.append(sum(r['tpot_ms'] for r in rows)/len(rows))
                    item = ab.distribution(wave_means)
                    groups.append(item)
                    ab.append_jsonl(folder/'warmup_groups.jsonl', {'cell': cell, 'group': group,
                        'unit': 'mean TPOT per whole wave', **item})
                    print(f'WARMUP {folder.name} {cell} ' + json.dumps(item), flush=True)
                    if admitted(groups):
                        break
                else:
                    raise ab.SafetyFailure('Warmup not stable for '+cell+'; no retry')
                ab.dump(folder/(cell+'-warmup.json'), {'groups': groups, 'admitted': True})

            if candidate:
                ab.dump(folder/'graph_policy'/'snapshot_warmup', {'excluded': True})
                await checked_wave(PROTOCOL['cells'][0], 'presteady_attestation_excluded', 0)
            for window in range(windows):
                for cell in PROTOCOL['cells']:
                    before = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                    began = time.perf_counter()
                    rows = []
                    for number in range(PROTOCOL['steady_waves_per_window_per_cell']):
                        rows.extend(await checked_wave(cell, 'steady', number, window))
                        if (number+1)%4 == 0:
                            print(f'STEADY {folder.name} {cell} w{window} waves={number+1}', flush=True)
                    elapsed = time.perf_counter()-began
                    after = ab.info_stats(await ab.api(session, base, '/server_info'), required=False)
                    item = ab.aggregate('full', cell, window, rows, before, after,
                        {'scope':'device_telemetry.csv; device-wide, not allocator peak'}, elapsed)
                    item.update(arm=ordinal, window=window, kernel_variant=variant, profiler=False,
                        diagnostic_hooks=False, waves=16, load=PROTOCOL['load'])
                    ab.append_jsonl(root/'summaries.jsonl', item)
                    summaries[(cell,window)] = item
                    print(f'WINDOW {folder.name} {cell} '+json.dumps({k:item[k] for k in
                        ('request_count','ttft_ms','tpot_ms','e2e_ms','client_observed_output_tokens_per_s','acceptance')}),flush=True)
            if candidate:
                ab.dump(folder/'graph_policy'/'snapshot_graph', {'excluded':True})
                await checked('p2048_c1',0,'attestation_excluded',0)
                # A zero steady delta is VALID no-effect evidence under a
                # multirow fallback policy. Do not require c1's positive gate.
                validate_graph_attestation(folder,candidate)
                policy=folder/'graph_policy'
                before=json.loads((policy/'graph_warmup.json').read_text())
                after=json.loads((policy/'graph_final.json').read_text())
                if set(after['actual_execute_by_bs']) != {'1'}:
                    raise ab.SafetyFailure('Graph replay outside original bs1 geometry')
                ab.dump(folder/'graph_window.json',{
                    'actual_execute_delta':after['actual_execute']-before['actual_execute'],
                    'fallback_delta':after['fallback']-before['fallback'],
                    'host_mirror_reads_delta':after['host_mirror_reads']-before['host_mirror_reads'],
                    'span':'all steady c4+c8 plus excluded presteady c4 wave; final snapshot before c1 attestation decode',
                    'actual_execute_by_bs':after['actual_execute_by_bs'],
                    'no_positive_steady_delta_required':True})
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
    integrity(template,args)
    return summaries


async def run(cli):
    inherited=json.loads((cli.inherit/'manifest.json').read_text())
    template=copy.deepcopy(inherited['execution_integrity_template'])
    args=SimpleNamespace(**template['args'])
    args.observe_work=False
    args.startup_timeout=min(args.startup_timeout,600)
    dataset=json.loads((Path(args.oracle_jsonl).parent/'dataset.json').read_text())
    refs=ab.load_frozen_references(Path(args.oracle_jsonl),dataset)
    integrity(template,args)
    cli.output.mkdir(parents=True,exist_ok=False)
    source_tools=tools_sha()
    began=time.time()
    ab.dump(cli.output/'manifest.json',{'protocol':PROTOCOL,'created_at':began,
        'inherit':str(cli.inherit),'inherit_sha256':sha(cli.inherit/'manifest.json'),
        'execution_integrity_template':template,'tool_sha256':source_tools,
        'experiment_worktree_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        'performance':'UNPROVEN'})
    status={'status':'running','performance_claim':'unproven'}
    ab.dump(cli.output/'status.json',status)

    async def execute():
        controls=await arm(args,template,refs,dataset,cli.output,0,'baseline',2)
        noise={cell:noise_gate(controls[(cell,0)],controls[(cell,1)]) for cell in PROTOCOL['cells']}
        ab.dump(cli.output/'control_noise.json',noise)
        if not all(item['admitted'] for item in noise.values()):
            raise ab.SafetyFailure('Concurrent baseline/baseline noise admission failed; no candidate timing')
        status['baseline_baseline_noise_admitted']=True
        arms={}
        for ordinal,variant in enumerate(PROTOCOL['order'],1):
            arms[ordinal]=await arm(args,template,refs,dataset,cli.output,ordinal,variant,1)
            if ordinal not in (2,4):
                continue
            failures=[]
            for cell in PROTOCOL['cells']:
                control,candidate=((arms[1][(cell,0)],arms[2][(cell,0)]) if ordinal==2
                                   else (arms[4][(cell,0)],arms[3][(cell,0)]))
                item={'cell':cell,'order':'AB' if ordinal==2 else 'BA',
                    'ratios_target_graph_over_baseline':ratios(control,candidate),
                    'throughput_ratio':candidate['client_observed_output_tokens_per_s']/control['client_observed_output_tokens_per_s']}
                item['e2e_gain_at_least_one_percent']=item['ratios_target_graph_over_baseline']['e2e_ms']['mean']<=.99
                item['p95_regressed_metrics']=[m for m,r in item['ratios_target_graph_over_baseline'].items() if r['p95']>1]
                item['p95_guard']='failed' if item['p95_regressed_metrics'] else 'passed'
                ab.append_jsonl(cli.output/'comparisons.jsonl',item)
                print('COMPARISON '+json.dumps(item),flush=True)
                if item['p95_regressed_metrics']:
                    failures.append({'cell':cell,'metrics':item['p95_regressed_metrics']})
            if failures:
                raise ab.SafetyFailure('p95 regression; stop remaining orders, restore safe OFF baseline: '+json.dumps(failures))
        drift={cell:noise_gate(arms[1][(cell,0)],arms[4][(cell,0)]) for cell in PROTOCOL['cells']}
        ab.dump(cli.output/'counterbalanced_control_drift.json',drift)
        if not all(d['admitted'] for d in drift.values()):
            raise ab.SafetyFailure('Concurrent counterbalanced control drift not admitted')
        status.update(status='counterbalanced_concurrency_completed',
            performance_claim='See both-order per-cell gains; no c1 extrapolation; no p95 regression')

    task=asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,task.cancel)
    try:
        await asyncio.wait_for(execute(),timeout=PROTOCOL['wall_budget_seconds'])
    except asyncio.CancelledError:
        status.update(status='safety_stopped',error='Supervisor stop')
    except Exception as error:
        status.update(status='safety_stopped',error_type=type(error).__name__,error=str(error))
    finally:
        try:
            integrity(template,args)
            if tools_sha()!=source_tools:
                raise ab.SafetyFailure('Tool sources changed during measurement')
            status['sources_and_frozen_references_unchanged']=True
        except Exception as error:
            status.update(status='safety_stopped',sources_and_frozen_references_unchanged=False,error=str(error))
        status.update(finished_at=time.time(),process_local_rollback='owned services stopped; serving baseline untouched')
        ab.dump(cli.output/'status.json',status)
    print(json.dumps(status),flush=True)
    return 0 if status['status']=='counterbalanced_concurrency_completed' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
