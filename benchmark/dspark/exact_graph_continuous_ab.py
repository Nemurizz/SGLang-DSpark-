"""Fixed continuous c4/c8/c16 serving, unchanged h58 inference candidate.

h59 short barrier-wave warmup failure is retained, not retried or discarded.
Load model follows SGLang bench_serving: each completion releases another
request without waiting for other clients. Frozen eight variants cycle at c16;
cache is OFF, so repeated variants do not introduce cache-hit benefits.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp
import dspark_prefix_ab as ab
import exact_graph_concurrent_ab as protocol_base
from exact_target_graph_ab import validate_graph_attestation
from target_graph_diagnostic import TargetGraphServer, start
from stable_baseline_gate import validate_row
from stable_cache_validation import wave
from steady_tpot import DeviceTelemetry, drain_listener, integrity


PROTOCOL = {**protocol_base.PROTOCOL,
    'cells':['p2048_c4','p2048_c8','p2048_c16'],
    'load':'continuous replenishment: fixed c clients, immediate next request per completion',
    'steady_requests_per_client':32,
    'steady_request_count_by_concurrency':{'4':128,'8':256,'16':512},
    'warmup':'three fixed blocks, each 8 completions/client; all retained/excluded',
    'warmup_fixed_groups':3,'warmup_requests_per_client_per_group':8,
    'warmup_gate_unit':'three balanced continuous-block mean TPOTs',
    'warmup_mean_spread_limit':.005,
    'warmup_within_block_percentiles_not_a_stationarity_gate':True,
    'oracle_mapping':'c4->h32 c4; c8->h32 c8; c16 slot%8->h32 c8 frozen tokens AND five acceptance',
    'closed_loop_tail':'startup and final drain included symmetrically; no steady-row discard',
    'predecessor':'h59 FAILED short barrier-wave warmup; retained as negative protocol evidence',
    'claim_scope':'2K/128 continuous capped c4,c8,c16 on current instance; not arrival-rate SLO saturation',
    'wall_budget_seconds':5400,
    'sources':['https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx',
               'https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx'],
}
for obsolete in ('steady_waves_per_window_per_cell','steady_requests_c4','steady_requests_c8',
                 'warmup_group_waves','warmup_min_groups','warmup_max_groups','warmup_wave_p95_over_p50_limit'):
    PROTOCOL.pop(obsolete,None)


def oracle_cell(cell):
    return 'p2048_c8' if cell=='p2048_c16' else cell


async def arm(args,template,refs,dataset,root,ordinal,variant,windows):
    integrity(template,args)
    folder=root/f'{ordinal:02d}_{variant}'
    folder.mkdir()
    args.output=str(folder)
    candidate=variant=='target_graph'
    server=TargetGraphServer(args,'full',folder,diagnostic=False,candidate=candidate)
    telemetry=DeviceTelemetry(folder)
    summaries={}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                connector=aiohttp.TCPConnector(limit=16,force_close=True)) as session:
            telemetry.begin()
            base=await start(server,session)
            validate_graph_attestation(folder,candidate)

            async def checked(cell,slot,phase,number,window=None):
                source=oracle_cell(cell)
                spec=dataset[source]
                v=slot%spec['concurrency']
                began=time.time()
                row=await ab.generate(session,base,spec['prompts'][v],128,args.seed)
                row.update(cell=cell,oracle_cell=source,variant=v,slot=slot,
                    concurrency=int(cell.rsplit('c',1)[1]),phase=phase,iteration=number,window=window,
                    kernel_variant=variant,arm=ordinal,began_wall_time=began,finished_wall_time=time.time())
                ab.append_jsonl(folder/'requests.jsonl',row)
                validate_row(row,spec,128,ordinary=False,require_cold=True)
                ab.check_reference(refs[(source,v)],row)
                if row['meta_info']['num_retractions']!=0:raise ab.SafetyFailure('Requires zero retractions')
                server.check()
                return row

            async def burst(cell,number):
                return await wave([checked(cell,s,'correctness_concurrent',number) for s in range(int(cell.rsplit('c',1)[1]))])

            async def load(cell,phase,count,window=None,group=None):
                c=int(cell.rsplit('c',1)[1])
                async def client(slot):
                    out=[]
                    for i in range(count):
                        row=await checked(cell,slot,phase,i if group is None else group*count+i,window)
                        out.append(row)
                    return out
                began=time.perf_counter()
                chunks=await wave([client(s) for s in range(c)])
                elapsed=time.perf_counter()-began
                rows=[r for chunk in chunks for r in chunk]
                record={'cell':cell,'phase':phase,'window':window,'group':group,'clients':c,
                    'requests_per_client':count,'requests':len(rows),'elapsed_s':elapsed,
                    'output_tokens_per_s':len(rows)*128/elapsed,
                    'mean_tpot_ms':sum(r['tpot_ms'] for r in rows)/len(rows)}
                ab.append_jsonl(folder/'load_blocks.jsonl',record)
                print('LOAD '+folder.name+' '+json.dumps(record),flush=True)
                return rows,elapsed

            for cell in PROTOCOL['correctness_cells']:
                for v in range(dataset[cell]['concurrency']):await checked(cell,v,'correctness_serial',v)
                for number in range(2):await burst(cell,number)
                print(f'CORRECTNESS {folder.name} {cell} serial+2bursts PASS',flush=True)
            for number in range(2):await burst('p2048_c16',number)
            ab.dump(folder/'correctness.json',{'passed':True,'requests':71,
                'frozen_tokens_and_five_acceptance':True,'c16_duplicate_variants_mapping':'slot%8'})

            for cell in PROTOCOL['cells']:
                means=[]
                for group in range(PROTOCOL['warmup_fixed_groups']):
                    rows,_=await load(cell,'warmup_excluded',8,group=group)
                    means.append(sum(r['tpot_ms'] for r in rows)/len(rows))
                spread=max(means)/min(means)-1
                ab.dump(folder/(cell+'-warmup.json'),{'means':means,'spread':spread,'admitted':spread<=.005,
                    'no_adaptive_retry':True,'unit':'balanced continuous block mean TPOT'})
                if spread>.005:
                    if not PROTOCOL.get('exploratory_descriptive_only',False):
                        raise ab.SafetyFailure('Fixed continuous warmup unstable '+cell)
                    print('NOT_QUALIFIED warmup '+cell+' spread='+str(spread),flush=True)
            if candidate:
                ab.dump(folder/'graph_policy'/'snapshot_warmup',{'excluded':True})
                await checked('p2048_c1',0,'presteady_attestation_excluded',0)
            for window in range(windows):
                for cell in PROTOCOL['cells']:
                    before=ab.info_stats(await ab.api(session,base,'/server_info'),required=False)
                    rows,elapsed=await load(cell,'steady',32,window)
                    after=ab.info_stats(await ab.api(session,base,'/server_info'),required=False)
                    item=ab.aggregate('full',cell,window,rows,before,after,
                        {'scope':'device_telemetry.csv; device-wide, not allocator peak'},elapsed)
                    item.update(arm=ordinal,window=window,kernel_variant=variant,profiler=False,
                        diagnostic_hooks=False,load=PROTOCOL['load'],requests_per_client=32)
                    ab.append_jsonl(root/'summaries.jsonl',item)
                    summaries[(cell,window)]=item
                    print('WINDOW '+folder.name+' '+cell+' '+json.dumps({k:item[k] for k in
                        ('request_count','ttft_ms','tpot_ms','e2e_ms','client_observed_output_tokens_per_s','acceptance')}),flush=True)
            if candidate:
                ab.dump(folder/'graph_policy'/'snapshot_graph',{'excluded':True})
                await checked('p2048_c1',0,'attestation_excluded',0)
                validate_graph_attestation(folder,candidate)
                policy=folder/'graph_policy'
                a=json.loads((policy/'graph_warmup.json').read_text())
                b=json.loads((policy/'graph_final.json').read_text())
                if set(b['actual_execute_by_bs'])!={'1'}:raise ab.SafetyFailure('Unexpected graph geometry')
                ab.dump(folder/'graph_window.json',{'actual_execute_delta':b['actual_execute']-a['actual_execute'],
                    'fallback_delta':b['fallback']-a['fallback'],
                    'host_mirror_reads_delta':b['host_mirror_reads']-a['host_mirror_reads'],
                    'span':'steady c4/c8/c16 plus one excluded presteady c1 request; snapshot before final attestation decode',
                    'actual_execute_by_bs':b['actual_execute_by_bs'],'zero_steady_graph_is_valid_no_effect_evidence':True})
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
    integrity(template,args)
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    # Swap only benchmark callbacks. Candidate worker hooks/sources unchanged.
    protocol_base.PROTOCOL=PROTOCOL
    protocol_base.arm=arm
    raise SystemExit(asyncio.run(protocol_base.run(parser.parse_args())))
