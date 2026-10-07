"""Read-only nsys SQLite + host trace decomposition by launch-correlated decode.

Kernel sums are GPU execution costs, NOT end-to-end latency. Host spans overlap
GPU execution. Wall minus thread CPU includes CUDA waits and descheduling.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics


def dist(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def q(f):
        i = (len(values)-1)*f
        lo, hi = int(i), min(len(values)-1, int(i)+1)
        return values[lo] + (values[hi]-values[lo])*(i-lo)
    return {'n': len(values), 'mean': statistics.mean(values), 'p50': q(.5), 'p95': q(.95)}


def analyze(folder):
    requests = [json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
    requests = [r for r in requests if r['phase'].startswith('diagnostic_')]
    ids = {r['meta_info']['id'] for r in requests}
    host = [json.loads(x) for x in (folder/'worker_trace.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in host:
        if row.get('rid') in ids:
            groups[row['method']].append(row)
    result = {'folder': str(folder), 'scope': 'profiled cause diagnosis only; NOT uninstrumented A/B',
        'requests': len(requests), 'host': {method: {
            'wall_ms': dist([x['wall_ns']/1e6 for x in rows]),
            'thread_cpu_ms': dist([x['thread_cpu_ns']/1e6 for x in rows]),
            'wall_minus_thread_cpu_ms': dist([(x['wall_ns']-x['thread_cpu_ns'])/1e6 for x in rows])}
            for method, rows in groups.items()},
        'sse_tpot_ms': dist([r['tpot_ms'] for r in requests]),
        'host_decode_spans': len(groups['_forward_decode']),
        'cache_method_calls': Counter(x['method'] for x in host if x['method'].startswith('cache_'))}
    result['cache_observer_coverage'] = ('actual UnifiedRadixCache' if any(x.get('tree_cache_class')=='UnifiedRadixCache' for x in host)
        else 'unavailable: initial observer wrapped legacy RadixCache, NOT used by current service')
    # Spec-v2 eager overlap executes one lookahead iteration beyond the 39
    # verifies reported on completed outputs. Keep it; don't count it as
    # acceptance or silently discard its execution cost.
    counts = Counter(x['rid'] for x in groups['_forward_decode'])
    assert set(counts.values()) == {40}, counts
    result['actual_forward_decode_calls_per_request'] = dict(counts)
    result['reported_verify_count_per_request'] = [r['meta_info']['spec_verify_ct'] for r in requests]
    result['host_between_decode_starts_ms'] = dist([
        (b['start_ns']-a['start_ns'])/1e6
        for rid in ids
        for a,b in zip(sorted([r for r in groups['_forward_decode'] if r['rid']==rid], key=lambda r:r['step']),
                       sorted([r for r in groups['_forward_decode'] if r['rid']==rid], key=lambda r:r['step'])[1:])])
    if not (folder/'trace.sqlite').exists():
        return result
    connection = sqlite3.connect('file:' + str(folder/'trace.sqlite') + '?mode=ro', uri=True)
    strings = dict(connection.execute('select id,value from StringIds'))
    ranges = defaultdict(list)
    for start,end,tid,label,textid in connection.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
        label = label or strings.get(textid, '')
        if label.startswith('DSpark._forward_decode '):
            details = json.loads(label.split(' ',1)[1])
            if details['rid'] in ids:
                ranges[tid].append({'start': start, 'end': end, **details})
    for values in ranges.values():
        values.sort(key=lambda x:x['start'])
    starts = {tid:[x['start'] for x in rows] for tid,rows in ranges.items()}
    def interval(tid, timestamp):
        if tid not in ranges:
            return None
        i = bisect.bisect_right(starts[tid], timestamp)-1
        if i < 0 or timestamp >= ranges[tid][i]['end']:
            return None
        return ranges[tid][i]
    launches, apis = {}, defaultdict(Counter)
    api_times = defaultdict(list)
    for start,end,tid,correlation,nameid in connection.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        span = interval(tid, start)
        if span:
            key = (span['rid'], span['step'])
            launches[(tid & ~0xffffff, correlation)] = key
            apis[key][strings[nameid]] += 1
            api_times[strings[nameid]].append((end-start)/1e6)
    kernel_groups = defaultdict(list)
    signature = defaultdict(Counter)
    kernel_by_step = defaultdict(list)
    for start,end,pid,corr,nameid,gx,gy,gz,bx,by,bz,smem in connection.execute(
            'select start,end,globalPid,correlationId,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL'):
        step = launches.get((pid, corr))
        if step is None:
            continue
        name = strings[nameid]
        key = (name,gx,gy,gz,bx,by,bz,smem)
        signature[step][key] += 1
        kernel_groups[name].append((end-start)/1e6)
        kernel_by_step[step].append((start,end))
    assert len(kernel_by_step) == len(requests)*40, 'Incomplete launch-correlation coverage'
    result['decode_kernel_count'] = sum(map(len,kernel_by_step.values()))
    result['kernels_per_decode'] = dist([len(v) for v in kernel_by_step.values()])
    result['gpu_kernel_sum_ms_per_decode'] = dist([sum(b-a for a,b in values)/1e6 for values in kernel_by_step.values()])
    result['gpu_execution_span_ms_per_decode'] = dist([(max(b for a,b in values)-min(a for a,b in values))/1e6 for values in kernel_by_step.values()])
    result['kernel_groups'] = sorted([{'name': name, 'count':len(values), 'sum_ms':sum(values), 'duration_ms':dist(values)}
                                      for name,values in kernel_groups.items()], key=lambda r:r['sum_ms'], reverse=True)
    result['runtime_api_in_decode'] = {name:{'count': len(values),'sum_ms':sum(values),'duration_ms':dist(values)}
                                       for name,values in api_times.items()}
    by_request = {}
    for req in requests:
        rid = req['meta_info']['id']
        tids = [(tid, values) for tid,values in ranges.items() if any(v['rid']==rid for v in values)]
        keys = [(rid, step) for step in range(1,41)]
        by_request[rid] = {'ordinal': req['ordinal'], 'mode': req['mode'], 'tpot_ms':req['tpot_ms'],
            'mean_kernel_sum_ms_per_actual_decode': statistics.mean(sum(b-a for a,b in kernel_by_step[key])/1e6 for key in keys)}
    # Second read to attribute GEMM durations by request without inferring from
    # host spans (host/GPU overlap and synchronizations move costs across spans).
    gemm_request = defaultdict(list)
    for start,end,pid,corr,nameid in connection.execute('select start,end,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        key = launches.get((pid,corr))
        if key and strings[nameid]=='matmul_kernel_persistent':
            gemm_request[key[0]].append((end-start)/1e6)
    for rid,item in by_request.items():
        item['mean_gemm_sum_ms_per_actual_decode'] = sum(gemm_request[rid])/40
    result['gpu_execution_by_request'] = by_request
    per_step = []
    for step in sorted({step for rid,step in signature}):
        counters = [signature[(rid,step)] for rid in sorted(ids)]
        assert all(counters[0]==c for c in counters), 'Within-mode launch geometry not identical'
        per_step.append({'step': step,'signature':[{'geometry':list(key),'count':value} for key,value in sorted(counters[0].items())]})
    result['per_step_launch_geometry'] = per_step
    connection.close()
    telemetry = list(csv.DictReader((folder/'device_telemetry.csv').open(), skipinitialspace=True))
    epochs = [(r['meta_info']['response_sent_to_client_ts']-r['e2e_ms']/1000,
               r['meta_info']['response_sent_to_client_ts']) for r in requests]
    from datetime import datetime
    sampled = [r for r in telemetry if any(a <= datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp() <= b for a,b in epochs)]
    result['telemetry_during_diagnostic_requests'] = {'sample_count':len(sampled),
        'columns':{k:dict(Counter(r[k] for r in sampled)) for k in ('pstate','clocks.current.sm [MHz]',
                  'clocks.current.memory [MHz]','clocks_event_reasons.active')},
        'power_watts': dist([float(r['power.draw [W]'].split()[0]) for r in sampled])}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous report')
    result = analyze(args.folder)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('kernel_groups','per_step_launch_geometry','runtime_api_in_decode')}, indent=2))
    print('TOP_KERNELS', json.dumps([(r['name'][:120],r['count'],r['sum_ms']) for r in result.get('kernel_groups',[])[:12]]))
